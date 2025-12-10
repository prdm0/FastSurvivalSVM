# R/tune.R

# ==============================================================================
#  PART 1: Custom Kernel Helpers & Metadata
# ==============================================================================

#' Helper to create variants of a custom kernel function for Tuning
#'
#' Generates a list of kernel functions with hardcoded parameters for use in
#' \code{\link{tune_fastsvm}}. This bridges the gap between R closures (functions
#' with memory) and Python's \code{GridSearchCV}.
#'
#' Unlike standard functions, the kernels generated here carry metadata 
#' (class \code{"fastsvm_custom_kernel"}) allowing them to be printed legibly
#' and their parameters accessed via the \code{$} operator.
#'
#' @param kernel_factory A function (factory) that takes numeric parameters
#'   as arguments and returns a kernel function with signature
#'   \code{function(x, z)}. See examples.
#' @param ... Vectors of parameters to expand (e.g., \code{a = c(1, 2)}).
#'
#' @return A named list of R functions. Each function has attributes 
#'   \code{"kernel_name"} and \code{"kernel_args"} for easy identification.
#'
#' @examples
#' # 1. Define a factory for a Wavelet kernel
#' #    Notice the use of force() to handle lazy evaluation.
#' make_wavelet <- function(a = 1) {
#'   force(a)
#'   function(x, z) {
#'     u <- (as.numeric(x) - as.numeric(z)) / a
#'     prod(cos(1.75 * u) * exp(-0.5 * u^2))
#'   }
#' }
#' 
#' # 2. Create variants with different 'a' values
#' k_list <- create_kernel_variants(make_wavelet, a = c(0.5, 1.0))
#' 
#' # 3. Inspect the list (names are descriptive)
#' print(names(k_list))
#' #> [1] "custom_a=0.5" "custom_a=1"
#' 
#' # 4. Access parameters directly using $
#' k <- k_list[[1]]
#' print(k)      # <FastSurvivalSVM Kernel: custom_a=0.5>
#' print(k$a)    # 0.5
#'
#' @export
create_kernel_variants <- function(kernel_factory, ...) {
  args_grid <- expand.grid(..., stringsAsFactors = FALSE)
  kernel_list <- list()

  for (i in 1:nrow(args_grid)) {
    current_args <- as.list(args_grid[i, , drop = FALSE])
    k_func <- do.call(kernel_factory, current_args)

    # Formatar nome limpo (ex: "a=1_b=2") para exibição
    fmt_args <- lapply(current_args, function(x) if(is.numeric(x)) round(x, 4) else x)
    name <- paste(names(fmt_args), fmt_args, sep = "=", collapse = "_")

    # Anexar metadados ao objeto função
    attr(k_func, "kernel_name") <- name
    attr(k_func, "kernel_args") <- current_args
    class(k_func) <- c("fastsvm_custom_kernel", class(k_func))

    # O nome na lista ajuda a identificar no grid search
    kernel_list[[name]] <- k_func
  }
  return(kernel_list)
}

#' Print method for custom kernels
#' @export
print.fastsvm_custom_kernel <- function(x, ...) {
  if (requireNamespace("cli", quietly = TRUE)) {
    cli::cli_text("{.cls FastSurvivalSVM Kernel}: {.strong {attr(x, 'kernel_name')}}")
  } else {
    cat(sprintf("<FastSurvivalSVM Kernel: %s>\n", attr(x, 'kernel_name')))
  }
  invisible(x)
}

#' Accessor for Custom Kernel Parameters
#'
#' Allows accessing hyperparameters of a custom kernel object using the \code{$} operator.
#' For example, if \code{k} is a kernel created with \code{a=1}, \code{k$a} returns 1.
#'
#' @param x A custom kernel function with class \code{"fastsvm_custom_kernel"}.
#' @param name The name of the parameter to extract (e.g., "a", "gamma").
#' @export
`$.fastsvm_custom_kernel` <- function(x, name) {
  attr(x, "kernel_args")[[name]]
}

# ==============================================================================
#  PART 2: Internal Helpers (Grid Reconstruction)
# ==============================================================================

# Helper to flatten nested lists returned by Python
.simplify_params <- function(params_list) {
  lapply(params_list, function(x) {
    if (is.list(x) && length(x) == 1 && is.null(names(x))) {
      return(x[[1]])
    }
    x
  })
}

# Emulate Sklearn ParameterGrid in R (ROBUST VERSION)
# This creates the exact list of candidate parameter sets that sklearn generated.
# By matching indices, we can retrieve the original R kernel object (with attributes)
# instead of the opaque Python wrapper returned by reticulate.
.flatten_sklearn_grid <- function(param_grid) {
  if (!is.null(names(param_grid))) param_grid <- list(param_grid)
  
  candidates <- list()
  
  for (block in param_grid) {
    # Sklearn sorts keys alphabetically
    keys <- sort(names(block))
    values_list <- block[keys]
    
    # 1. Expand INDICES instead of objects to avoid coercion issues
    indices_list <- lapply(values_list, function(x) seq_along(x))
    
    # 2. Match sklearn order (vary last element fastest) -> R varies first fastest
    # So we expand reversed indices, then reverse columns back
    rev_inds <- rev(indices_list)
    grid_inds <- do.call(expand.grid, c(rev_inds, KEEP.OUT.ATTRS = FALSE))
    
    if (nrow(grid_inds) > 0) {
      grid_inds <- grid_inds[, rev(names(grid_inds)), drop = FALSE]
      
      # 3. Reconstruct list of parameter sets using indices
      rows <- lapply(seq_len(nrow(grid_inds)), function(i) {
        row_list <- list()
        for (k in keys) {
          idx <- grid_inds[i, k]
          row_list[[k]] <- values_list[[k]][[idx]]
        }
        row_list
      })
      candidates <- c(candidates, rows)
    }
  }
  candidates
}

# ==============================================================================
#  PART 3: Tuning Functions
# ==============================================================================

#' Single Grid Search for FastKernelSurvivalSVM (Internal Wrapper)
#'
#' A wrapper around \code{sklearn.model_selection.GridSearchCV} to tune a
#' single kernel configuration (estimator). Uses \code{cli} for elegant output.
#'
#' @section Parallelization Guidelines:
#' The \code{n_jobs} parameter controls parallelization within Python's GridSearchCV.
#' \itemize{
#'   \item \strong{Built-in Kernels}: If you use standard kernels passed as strings 
#'         (e.g., \code{"rbf"}, \code{"linear"}), you can safely use parallel processing. 
#'         Set \code{n_jobs = parallel::detectCores()} or any integer > 1.
#'   \item \strong{Custom R Kernels}: If you pass \strong{R functions} as kernels, 
#'         you \strong{MUST} set \code{n_jobs = 1}. Python's \code{joblib} cannot 
#'         serialize (pickle) R functions to worker processes. Using \code{n_jobs > 1} 
#'         will cause the process to hang or crash with serialization errors.
#' }
#'
#' @inheritParams tune_random_machines
#' @param param_grid A named list (or list of lists) of parameters to tune.
#'   Example: \code{list(alpha = c(0.1, 1), rank_ratio = c(0, 1))}.
#' @param refit Logical. If \code{TRUE}, refit model with best params on full data.
#'
#' @return An object of class \code{"fastsvm_grid"} containing:
#'   \itemize{
#'     \item \code{best_params}: A named list of the optimal parameters. If a custom
#'           kernel was selected, this will be the R function object (use \code{$param} to access values).
#'     \item \code{best_score}: The mean cross-validated concordance index.
#'     \item \code{cv_results}: A data frame summarizing results for all grid combinations.
#'   }
#'
#' @examples
#' \dontrun{
#' if (reticulate::py_module_available("sksurv")) {
#'   library(FastSurvivalSVM)
#'   set.seed(42)
#'   df <- data_generation(n = 200, prop_cen = 0.3)
#'
#'   # --- Example 1: Tuning Standard RBF (Parallel) ---
#'   # Native strings allow full parallelization.
#'   grid_rbf <- list(
#'     kernel = "rbf",
#'     alpha  = c(0.01, 0.1, 1),
#'     gamma  = c(0.01, 0.1)
#'   )
#'   
#'   res_rbf <- tune_fastsvm(
#'     data = df, time_col = "tempo", delta_col = "cens",
#'     param_grid = grid_rbf, cv = 3, 
#'     n_jobs = parallel::detectCores(), # Safe!
#'     verbose = 1
#'   )
#'   print(res_rbf)
#'
#'   # --- Example 2: Tuning Custom Kernel (Serial) ---
#'   # 1. Define Factory
#'   make_wav <- function(a=1) { force(a); function(x,z) {
#'      u<-(as.numeric(x)-as.numeric(z))/a; prod(cos(1.75*u)*exp(-0.5*u^2))
#'   }}
#'   
#'   # 2. Create Variants
#'   wav_vars <- create_kernel_variants(make_wav, a=c(0.5, 2.0))
#'   
#'   grid_wav <- list(
#'     kernel = wav_vars,
#'     alpha = c(0.1, 1.0)
#'   )
#'   
#'   # 3. Tune (n_jobs=1 mandatory)
#'   res_wav <- tune_fastsvm(
#'     data = df, time_col = "tempo", delta_col = "cens",
#'     param_grid = grid_wav, cv = 3, 
#'     n_jobs = 1, # Must be 1 for R functions
#'     verbose = 1
#'   )
#'   
#'   # 4. Accessing the chosen 'a' parameter
#'   best_k <- res_wav$best_params$kernel
#'   cat("Best 'a':", best_k$a, "\n") # Access via $ works!
#' }
#' }
#'
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
  # --- Input Validation ---
  if (!is.data.frame(data)) stop("Argument `data` must be a data.frame.", call. = FALSE)
  if (!time_col %in% names(data)) stop(sprintf("Column '%s' not found.", time_col))
  if (!delta_col %in% names(data)) stop(sprintf("Column '%s' not found.", delta_col))

  # --- Setup CLI & Emojis ---
  has_cli <- requireNamespace("cli", quietly = TRUE)
  has_emo <- requireNamespace("emo", quietly = TRUE)
  ji <- function(x, f="") if(has_emo) emo::ji(x) else f
  
  if (verbose > 0 && has_cli) {
    cli::cli_h2(paste(ji("gear"), "Starting Grid Search Tuning"))
    cli::cli_alert_info("CV Folds: {cv} | Parallel Jobs: {n_jobs}")
  }

  # --- Imports ---
  sklearn_sel <- reticulate::import("sklearn.model_selection", delay_load = TRUE)
  sksvm_mod   <- reticulate::import("sksurv.svm", delay_load = TRUE)
  sksurv_util <- reticulate::import("sksurv.util", delay_load = TRUE)
  try(reticulate::py_run_string("import warnings; warnings.simplefilter('ignore')"), silent = TRUE)
  
  # --- Data Prep ---
  x_cols <- setdiff(names(data), c(time_col, delta_col))
  X_mat  <- as.matrix(data[, x_cols, drop = FALSE])
  y_surv <- sksurv_util$Surv$from_arrays(
    event = as.logical(data[[delta_col]]), 
    time  = as.numeric(data[[time_col]])
  )

  # --- Estimator ---
  fixed_args <- list(...)
  for (p in c("max_iter", "degree", "random_state", "verbose")) {
    if (!is.null(fixed_args[[p]])) fixed_args[[p]] <- as.integer(fixed_args[[p]])
  }
  estimator <- do.call(sksvm_mod$FastKernelSurvivalSVM, fixed_args)

  # --- Grid Sanitization ---
  sanitize_single <- function(block) {
    lapply(block, function(val) {
      if (is.function(val)) return(list(val))
      if (is.list(val) && length(val) > 0) {
        if (is.function(val[[1]]) || inherits(val[[1]], "fastsvm_custom_kernel")) {
          return(unname(val)) # Fix for named lists of functions
        }
      }
      if (!is.list(val) && !is.vector(val)) return(list(val))
      if (length(val) == 1 && !is.list(val)) return(list(val))
      as.list(val)
    })
  }
  is_composite <- is.list(param_grid) && is.null(names(param_grid))
  clean_grid <- if (is_composite) lapply(param_grid, sanitize_single) else sanitize_single(param_grid)

  # --- Fit ---
  gs_instance <- sklearn_sel$GridSearchCV(
    estimator = estimator, param_grid = clean_grid, cv = as.integer(cv),
    n_jobs = if (is.null(n_jobs)) NULL else as.integer(n_jobs),
    verbose = as.integer(0),
    refit = refit
  )

  if (verbose > 0 && has_cli) cli::cli_progress_step("Fitting models...", spinner = TRUE)
  
  tryCatch({
    gs_instance$fit(X_mat, y_surv)
  }, error = function(e) {
    if (grepl("pickle", e$message, ignore.case = TRUE)) {
      stop("Pickle Error: Set 'n_jobs = 1' when using custom R kernels.", call. = FALSE)
    } else {
      stop(e)
    }
  })

  # --- Result Reconstruction (The Fix) ---
  best_idx   <- tryCatch(as.integer(gs_instance$best_index_) + 1L, error = function(e) NULL)
  best_score <- tryCatch(as.numeric(gs_instance$best_score_), error = function(e) NA)
  
  # Rebuild candidate list in R
  r_candidates <- .flatten_sklearn_grid(clean_grid)
  
  # Get Best Params (Original R Objects)
  best_params <- if (!is.null(best_idx) && best_idx <= length(r_candidates)) {
    r_candidates[[best_idx]]
  } else {
    tryCatch(reticulate::py_to_r(gs_instance$best_params_), error = function(e) NULL)
  }
  
  best_params <- .simplify_params(best_params)

  # --- CV Results Table ---
  py_res <- tryCatch(reticulate::py_to_r(gs_instance$cv_results_), error = function(e) NULL)
  
  cv_results <- NULL
  if (!is.null(py_res)) {
    params_df <- do.call(rbind, lapply(r_candidates, function(row) {
      lapply(row, function(val) {
        if (inherits(val, "fastsvm_custom_kernel")) return(attr(val, "kernel_name"))
        if (is.function(val)) return("<function>")
        val
      })
    }))
    params_df <- as.data.frame(params_df, stringsAsFactors = FALSE)
    # Simplify columns
    params_df[] <- lapply(params_df, function(x) if(is.list(x)) unlist(x) else x)
    colnames(params_df) <- paste0("param_", colnames(params_df))
    
    scores_df <- data.frame(
      mean_test_score = as.numeric(py_res$mean_test_score),
      std_test_score  = as.numeric(py_res$std_test_score),
      rank_test_score = as.integer(py_res$rank_test_score)
    )
    cv_results <- cbind(scores_df, params_df)
  }

  if (verbose > 0 && has_cli) {
    cli::cli_progress_done()
    cli::cli_alert_success("Tuning complete. Best C-index: {.val {round(best_score, 4)}}")
  }

  structure(
    list(
      grid_search_obj = gs_instance,
      best_estimator  = gs_instance$best_estimator_, 
      best_params     = best_params,
      best_score      = best_score,
      cv_results      = cv_results,
      x_cols = x_cols, time_col = time_col, delta_col = delta_col
    ),
    class = "fastsvm_grid"
  )
}

#' Multi-Kernel Tuning for Random Machines
#'
#' Orchestrates hyperparameter tuning for multiple kernels simultaneously.
#' This function allows you to define a list of kernel configurations (similar to
#' \code{\link{random_machines}}) and a corresponding list of parameter grids,
#' finding the best hyperparameters for each kernel type.
#'
#' @section Parallelization Guidelines:
#' \itemize{
#'   \item Use \code{n_jobs = parallel::detectCores()} if ALL kernels in the list are built-in strings (e.g., "rbf", "linear").
#'   \item Use \code{n_jobs = 1} if ANY kernel in the list is a custom R function.
#' }
#'
#' @param data A \code{data.frame} containing training data.
#' @param time_col Name of the time column.
#' @param delta_col Name of the event column (1=event, 0=censored).
#' @param kernel_mix A named list defining the base configuration for each kernel.
#'   Each element should be a list with at least the \code{kernel} parameter.
#' @param param_grids A named list of parameter grids corresponding to \code{kernel_mix}.
#'   The names must match the names in \code{kernel_mix}. Each element is a list
#'   of parameters to vary.
#' @param cv Number of cross-validation folds (default: 5).
#' @param n_jobs Number of parallel jobs for grid search. See guidelines.
#' @param verbose Verbosity level (0 or 1).
#' @param ... Additional fixed parameters passed to all estimators.
#'
#' @return An object of class \code{"random_machines_tune"}, containing the tuning
#'   results for each kernel.
#'
#' @examples
#' \dontrun{
#' if (reticulate::py_module_available("sksurv")) {
#'   library(FastSurvivalSVM)
#'   set.seed(42)
#'   df <- data_generation(n = 200, prop_cen = 0.3)
#'
#'   # --- 1. Define Base Kernels ---
#'   make_wav <- function(a=1) { force(a); function(x,z) {
#'      u<-(as.numeric(x)-as.numeric(z))/a; prod(cos(1.75*u)*exp(-0.5*u^2))
#'   }}
#'   
#'   base_kernels <- list(
#'     linear  = list(kernel = "linear"),
#'     rbf     = list(kernel = "rbf"),
#'     wavelet = list(kernel = make_wav(a=1)) # Initial placeholder
#'   )
#'
#'   # --- 2. Define Grids for Tuning ---
#'   # For custom kernels, use create_kernel_variants in the grid!
#'   wav_vars <- create_kernel_variants(make_wav, a=c(0.5, 1.0, 2.0))
#'   
#'   tuning_grids <- list(
#'     linear  = list(alpha = c(0.1, 1, 10)),
#'     rbf     = list(alpha = c(0.1, 1), gamma = c(0.01, 0.1)),
#'     wavelet = list(alpha = c(0.1, 1), kernel = wav_vars) # Tune 'a' via kernel variants
#'   )
#'
#'   # --- 3. Run Tuning ---
#'   # n_jobs=1 because we have a custom wavelet kernel
#'   results <- tune_random_machines(
#'     data = df, time_col = "tempo", delta_col = "cens",
#'     kernel_mix = base_kernels,
#'     param_grids = tuning_grids,
#'     cv = 3, n_jobs = 1, verbose = 1
#'   )
#'
#'   # --- 4. Inspect Results ---
#'   print(results)
#'   
#'   # Access best params
#'   print(results$rbf$best_params)
#'   
#'   # Access best 'a' for wavelet
#'   best_wav <- results$wavelet$best_params$kernel
#'   cat("Best wavelet 'a':", best_wav$a, "\n")
#' }
#' }
#' 
#' @export
tune_random_machines <- function(
  data,
  time_col  = "t",
  delta_col = "delta",
  kernel_mix,
  param_grids,
  cv        = 5L,
  n_jobs    = 1L,
  verbose   = 0L,
  ...
) {
  # Validations
  if (!is.data.frame(data)) stop("Argument `data` must be a data.frame.", call. = FALSE)
  if (!is.list(kernel_mix) || is.null(names(kernel_mix))) stop("`kernel_mix` must be a named list.")
  if (!is.list(param_grids) || is.null(names(param_grids))) stop("`param_grids` must be a named list.")
  
  common_names <- intersect(names(kernel_mix), names(param_grids))
  if (length(common_names) == 0) stop("No matching names between `kernel_mix` and `param_grids`.")

  # UI Setup
  has_cli <- requireNamespace("cli", quietly = TRUE)
  has_emo <- requireNamespace("emo", quietly = TRUE)
  ji <- function(x, f="") if(has_emo) emo::ji(x) else f

  results <- list()
  
  if (verbose > 0 && has_cli) {
    cli::cli_h1(paste(ji("rocket"), "Multi-Kernel Tuning Session"))
    cli::cli_alert_info("Optimizing {length(common_names)} kernel configurations.")
    cli::cli_rule()
  }
  
  for (kname in common_names) {
    base_def <- kernel_mix[[kname]]
    grid_def <- param_grids[[kname]]
    
    if (verbose > 0 && has_cli) {
      cli::cli_h2(paste(ji("mag"), "Tuning: {.strong {kname}}"))
    }
    
    # Prepare arguments for this specific kernel fit
    # We take the base definition (e.g. kernel="rbf") and pass it as fixed args
    # But we remove any key that is present in the grid (so grid takes precedence)
    fixed_args <- base_def
    for (p in names(grid_def)) fixed_args[[p]] <- NULL
    
    res <- tryCatch({
      do.call(tune_fastsvm, c(
        list(
          data       = data,
          time_col   = time_col,
          delta_col  = delta_col,
          param_grid = grid_def,
          cv         = cv,
          n_jobs     = n_jobs,
          verbose    = 0, # Inner verbose off
          refit      = TRUE
        ),
        fixed_args, # Pass remaining base params (e.g. kernel="rbf")
        list(...)   # Pass global extra params
      ))
    }, error = function(e) {
      if (has_cli) cli::cli_alert_danger("Failed: {e$message}")
      NULL
    })
    
    if (verbose > 0 && !is.null(res) && has_cli) {
      cli::cli_alert_success("Best C-index: {.val {round(res$best_score, 4)}}")
    }
    results[[kname]] <- res
  }
  
  structure(results, class = "random_machines_tune")
}

# -------------------------------------------------------------------
# S3 Methods
# -------------------------------------------------------------------

#' Print method for Random Machines tuning results
#' @export
print.random_machines_tune <- function(x, ...) {
  if (requireNamespace("cli", quietly = TRUE)) {
    cli::cli_h1("Random Machines Tuning Summary")
    
    df <- data.frame(
      Kernel = names(x),
      C_Index = sapply(x, function(mod) if(is.null(mod)) NA else mod$best_score),
      Status = sapply(x, function(mod) if(is.null(mod)) "Failed" else "OK")
    )
    df <- df[order(df$C_Index, decreasing = TRUE), ]
    print(df, row.names = FALSE)
    
  } else {
    cat("\nRandom Machines Tuning Summary\n")
    print(x) 
  }
  invisible(x)
}

#' Print method for single tuning result
#' @export
print.fastsvm_grid <- function(x, ...) {
  if (requireNamespace("cli", quietly = TRUE)) {
    cli::cli_div(theme = list(span.emph = list(color = "blue")))
    cli::cli_h2("Grid Search Result")
    cli::cli_alert_success("Best C-Index: {.val {round(x$best_score, 4)}}")
    
    cli::cli_text("{.strong Best Parameters:}")
    cli::cli_ul()
    
    params <- x$best_params
    for (nm in names(params)) {
      val <- params[[nm]]
      val_print <- if (inherits(val, "fastsvm_custom_kernel")) {
        attr(val, "kernel_name") 
      } else {
        as.character(val)
      }
      cli::cli_li("{.emph {nm}}: {val_print}")
    }
    cli::cli_end()
    
  } else {
    cat(sprintf("Best C-Index: %.4f\n", x$best_score))
    print(x$best_params)
  }
  invisible(x)
}