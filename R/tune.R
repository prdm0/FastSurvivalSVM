# R/gridsearch.R

#' Helper to create variants of a custom kernel function
#'
#' Generates a list of kernel functions with hardcoded parameters for tuning.
#' This bridges the gap between R closures and Python's GridSearchCV.
#'
#' @param kernel_factory A function that takes numeric parameters and returns a
#'   kernel function of signature \code{function(x, z)}.
#' @param ... Vectors of parameters to expand (e.g., a = c(1, 2)).
#'
#' @return A named list of kernel functions ready for \code{tune_fastsvm}.
#' @export
create_kernel_variants <- function(kernel_factory, ...) {
  args_grid <- expand.grid(..., stringsAsFactors = FALSE)
  kernel_list <- list()

  for (i in 1:nrow(args_grid)) {
    current_args <- as.list(args_grid[i, , drop = FALSE])
    # Cria a função (closure) com os parâmetros fixados
    k_func <- do.call(kernel_factory, current_args)

    # Cria um nome descritivo para o relatório (ex: custom_a=1_b=2)
    param_str <- paste(names(current_args), current_args, sep = "=", collapse = "_")
    name <- paste0("custom_", param_str)

    kernel_list[[name]] <- k_func
  }
  return(kernel_list)
}

#' Single Grid Search for FastKernelSurvivalSVM (Internal)
#'
#' Wrapper for \code{sklearn.model_selection.GridSearchCV}.
#'
#' @inheritParams tune_fastsvm_all
#' @param param_grid A list (or list of lists) of parameters to tune.
#' @export
tune_fastsvm <- function(
  data,
  time_col   = "t",
  delta_col  = "delta",
  param_grid,
  cv         = 5L,
  n_jobs     = 1L,
  verbose    = 0L,
  refit      = TRUE,
  ...
) {
  # --- Imports ---
  sklearn_sel <- reticulate::import("sklearn.model_selection", delay_load = TRUE)
  sksvm_mod   <- reticulate::import("sksurv.svm", delay_load = TRUE)
  sksurv_util <- reticulate::import("sksurv.util", delay_load = TRUE)
  
  # --- SILENCE PYTHON WARNINGS ---
  # Isso é crítico para evitar o flood de "ConvergenceWarning" no console do R
  py_warnings <- reticulate::import("warnings")
  py_warnings$filterwarnings("ignore") 
  # Se quiser ser mais específico:
  # py_warnings$filterwarnings("ignore", category = reticulate::import("sklearn.exceptions")$ConvergenceWarning)

  # --- Data Prep ---
  if (!time_col %in% names(data)) stop(sprintf("Column '%s' not found.", time_col))
  if (!delta_col %in% names(data)) stop(sprintf("Column '%s' not found.", delta_col))
  
  x_cols <- setdiff(names(data), c(time_col, delta_col))
  X_mat  <- as.matrix(data[, x_cols, drop = FALSE])
  y_surv <- sksurv_util$Surv$from_arrays(
    event = as.logical(data[[delta_col]]), 
    time  = as.numeric(data[[time_col]])
  )

  # --- Fixed Args Sanitization ---
  fixed_args <- list(...)
  int_params_to_check <- c("max_iter", "degree", "random_state", "verbose")
  for (p in int_params_to_check) {
    if (!is.null(fixed_args[[p]])) fixed_args[[p]] <- as.integer(fixed_args[[p]])
  }
  estimator <- do.call(sksvm_mod$FastKernelSurvivalSVM, fixed_args)

  # --- Grid Sanitization ---
  sanitize_single_grid <- function(grid_block) {
    lapply(grid_block, function(val) {
      if (is.function(val)) return(list(val))
      if (is.list(val) && length(val) > 0 && is.function(val[[1]])) return(unname(val))
      if (!is.list(val) && !is.vector(val)) return(list(val))
      if (length(val) == 1 && !is.list(val)) return(list(val))
      as.list(val)
    })
  }

  is_composite <- is.list(param_grid) && is.null(names(param_grid))
  clean_grid <- if (is_composite) lapply(param_grid, sanitize_single_grid) else sanitize_single_grid(param_grid)

  # --- Fit ---
  gs_instance <- sklearn_sel$GridSearchCV(
    estimator = estimator, param_grid = clean_grid, cv = as.integer(cv),
    n_jobs = if (is.null(n_jobs)) NULL else as.integer(n_jobs),
    verbose = as.integer(verbose), refit = refit
  )

  if (verbose > 0) message("Starting GridSearchCV fitting...")
  
  tryCatch({
    # Executa o fit
    gs_instance$fit(X_mat, y_surv)
  }, error = function(e) {
    if (grepl("pickle", e$message, ignore.case = TRUE)) {
      stop("Pickle/Serialization Error: Set 'n_jobs = 1' when using custom R kernels.", call. = FALSE)
    } else {
      stop(e)
    }
  })

  # --- Results ---
  best_params <- tryCatch(reticulate::py_to_r(gs_instance$best_params_), error = function(e) NULL)
  best_score  <- tryCatch(as.numeric(gs_instance$best_score_), error = function(e) NA)
  
  cv_results <- tryCatch({
    res <- reticulate::py_to_r(gs_instance$cv_results_)
    res$params <- NULL 
    res <- lapply(res, function(col) {
      if (is.list(col) || inherits(col, "numpy.ndarray")) {
        tryCatch(as.vector(col), error = function(e) as.character(col))
      } else { col }
    })
    as.data.frame(res)
  }, error = function(e) NULL)

  structure(
    list(
      grid_search_obj = gs_instance,
      best_estimator  = gs_instance$best_estimator_, 
      best_params     = best_params,
      best_score      = best_score,
      cv_results      = cv_results,
      x_cols          = x_cols, time_col = time_col, delta_col = delta_col
    ),
    class = "fastsvm_grid"
  )
}

#' Multi-Kernel Grid Search (Tuning All Kernels)
#'
#' Runs a separate Grid Search for each kernel configuration provided in a named list.
#' Returns the best parameters for \strong{each} kernel type.
#'
#' @param data A \code{data.frame} with training data.
#' @param time_col Name of the time column.
#' @param delta_col Name of the event column.
#' @param grids A \strong{named list} of parameter grids. Each element should be a grid 
#'   dictionary for a specific kernel.
#' @param cv Integer. Number of folds.
#' @param n_jobs Integer. Use 1 if using custom kernels.
#' @param verbose Integer.
#' @param ... Additional fixed parameters (e.g., \code{max_iter}).
#'
#' @return An object of class \code{"fastsvm_multi_grid"}, which is a list of \code{fastsvm_grid} objects.
#'
#' @examples
#' \dontrun{
#' if (reticulate::py_module_available("sksurv")) {
#'   library(FastSurvivalSVM)
#'   set.seed(42)
#'   df <- data_generation(n = 150, prop_cen = 0.3)
#'
#'   # --- A. Custom Kernel Factories ---
#'   wavelet_factory <- function(a = 1.0) {
#'     force(a)
#'     function(x, z) {
#'       u <- (as.numeric(x) - as.numeric(z)) / a
#'       prod(cos(1.75 * u) * exp(-0.5 * u^2))
#'     }
#'   }
#'   
#'   cauchy_factory <- function(gamma = 1.0) {
#'     force(gamma)
#'     function(x, z) {
#'       d2 <- sum((as.numeric(x) - as.numeric(z))^2)
#'       1 / (1 + gamma * d2)
#'     }
#'   }
#'
#'   # --- B. Generate Variants ---
#'   wavelet_vars <- create_kernel_variants(wavelet_factory, a = c(0.5, 1.0, 2.0))
#'   cauchy_vars  <- create_kernel_variants(cauchy_factory, gamma = c(0.1, 1.0))
#'
#'   # --- C. Define Named Grids for All Kernels ---
#'   all_grids <- list(
#'     linear = list(kernel = "linear", alpha = c(0.1, 1.0), rank_ratio = 0.0),
#'     rbf = list(kernel = "rbf", alpha = c(0.1, 1.0), gamma = c(0.01, 0.1), rank_ratio = 0.0),
#'     wavelet = list(kernel = wavelet_vars, alpha = c(0.1, 1.0), rank_ratio = 0.0),
#'     cauchy = list(kernel = cauchy_vars, alpha = c(0.1, 1.0), rank_ratio = 0.0)
#'   )
#'
#'   # --- D. Run Tune All ---
#'   # As mensagens de convergência do Python serão silenciadas agora.
#'   results_all <- tune_fastsvm_all(
#'     data       = df,
#'     time_col   = "tempo",
#'     delta_col  = "cens",
#'     grids      = all_grids,
#'     cv         = 3,
#'     n_jobs     = 1,
#'     verbose    = 0,
#'     max_iter   = 500L
#'   )
#'
#'   print(results_all)
#' }
#' }
#' 
#' @export
tune_fastsvm_all <- function(
  data,
  time_col  = "t",
  delta_col = "delta",
  grids,
  cv        = 5L,
  n_jobs    = 1L,
  verbose   = 0L,
  ...
) {
  if (!is.list(grids) || is.null(names(grids))) {
    stop("`grids` must be a named list of parameter grids (one for each kernel).")
  }
  
  # --- SILENCE PYTHON WARNINGS (Global Scope for this call) ---
  # Garantia extra caso tune_fastsvm seja chamado diretamente
  tryCatch({
      py_warnings <- reticulate::import("warnings")
      py_warnings$filterwarnings("ignore")
  }, error = function(e) NULL)

  kernel_names <- names(grids)
  results <- list()
  
  if (verbose > 0) cat("Starting Multi-Kernel Tuning...\n")
  
  for (kname in kernel_names) {
    if (verbose > 0) cat(sprintf(">> Tuning Kernel: %s ... ", kname))
    
    # Executa o tuning individual para este kernel
    res <- tryCatch({
      tune_fastsvm(
        data       = data,
        time_col   = time_col,
        delta_col  = delta_col,
        param_grid = grids[[kname]],
        cv         = cv,
        n_jobs     = n_jobs,
        verbose    = 0, # Silencia o grid interno
        refit      = TRUE,
        ...
      )
    }, error = function(e) {
      warning(sprintf("Failed to tune kernel '%s': %s", kname, e$message))
      NULL
    })
    
    if (verbose > 0 && !is.null(res)) cat(sprintf("Done. Best Score: %.4f\n", res$best_score))
    results[[kname]] <- res
  }
  
  structure(results, class = "fastsvm_multi_grid")
}

# -------------------------------------------------------------------
# S3 Methods for fastsvm_multi_grid
# -------------------------------------------------------------------

#' Print method for multi-kernel results
#' @export
print.fastsvm_multi_grid <- function(x, ...) {
  cat("\nMulti-Kernel FastSurvivalSVM Tuning Results\n")
  cat("===========================================\n")
  
  # Tabela resumo
  summary_df <- data.frame(
    Kernel = names(x),
    Best_C_Index = sapply(x, function(mod) if(is.null(mod)) NA else mod$best_score),
    Status = sapply(x, function(mod) if(is.null(mod)) "Failed" else "OK")
  )
  
  # Ordenar por C-Index decrescente
  summary_df <- summary_df[order(summary_df$Best_C_Index, decreasing = TRUE), ]
  print(summary_df, row.names = FALSE)
  
  invisible(x)
}