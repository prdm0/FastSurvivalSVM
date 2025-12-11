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
#' #> [1] "a=0.5" "a=1"
#' 
#' # 4. Access parameters directly using $
#' k <- k_list[[1]]
#' print(k)      # <FastSurvivalSVM Kernel: a=0.5>
#' print(k$a)    # 0.5
#'
#' @export
create_kernel_variants <- function(kernel_factory, ...) {
  args_grid <- expand.grid(..., stringsAsFactors = FALSE)
  kernel_list <- list()

  for (i in 1:nrow(args_grid)) {
    current_args <- as.list(args_grid[i, , drop = FALSE])
    k_func <- do.call(kernel_factory, current_args)

    # Format clean name (e.g., "a=1_b=2") for display
    fmt_args <- lapply(current_args, function(x) if(is.numeric(x)) round(x, 4) else x)
    name <- paste(names(fmt_args), fmt_args, sep = "=", collapse = "_")

    # Attach metadata
    attr(k_func, "kernel_name") <- name
    attr(k_func, "kernel_args") <- current_args
    class(k_func) <- c("fastsvm_custom_kernel", class(k_func))

    # The name in the list helps identification in grid search
    kernel_list[[name]] <- k_func
  }
  return(kernel_list)
}

#' Print method for custom kernels
#' 
#' @param x An object of class \code{"fastsvm_custom_kernel"}.
#' @param ... Additional arguments passed to methods.
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
# Creates the exact list of candidate parameter sets that sklearn would generate.
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

#' Single Grid Search for FastKernelSurvivalSVM
#'
#' Executes a grid search to optimize hyperparameters for a single kernel configuration.
#' Uses cross-validation to evaluate the C-index.
#'
#' @section Parallelization Strategy:
#' The \code{cores} argument controls how computation is distributed:
#' \itemize{
#'   \item \strong{Standard Kernels ("rbf", "linear", etc.):} If the kernel is a string,
#'         the function uses Python's `GridSearchCV` native parallelism (via `joblib`).
#'         This is extremely fast and efficient.
#'         
#'   \item \strong{Custom R Kernels:} If the kernel is an R function, `scikit-learn`
#'         cannot parallelize natively (pickle error). In this case, the function
#'         automatically detects it and uses the \pkg{mirai} package to distribute
#'         grid candidates across parallel R workers. Each worker runs cross-validation
#'         for one candidate serially.
#' }
#'
#' @inheritParams tune_random_machines
#' @param param_grid A named list (or list of lists) of parameters to tune.
#'   Example: \code{list(alpha = c(0.1, 1), rank_ratio = c(0, 1))}.
#' @param refit Logical. If \code{TRUE}, refit model with best params on full data.
#'
#' @return An object of class \code{"fastsvm_grid"} containing:
#'   \itemize{
#'     \item \code{best_params}: A named list of optimal parameters.
#'     \item \code{best_score}: The best mean cross-validated C-index.
#'     \item \code{cv_results}: A data frame summarizing the search history.
#'     \item \code{best_estimator}: The fitted model object (if \code{refit=TRUE}).
#'   }
#'
#' @examples
#' \dontrun{
#' if (reticulate::py_module_available("sksurv") && requireNamespace("mirai", quietly = TRUE)) {
#'   library(FastSurvivalSVM)
#'   
#'   # --- Data Generation ---
#'   set.seed(42)
#'   df <- data_generation(n = 200, prop_cen = 0.3)
#'
#'   # =========================================================================
#'   # CASE 1: Standard Kernel (RBF) - Uses Python Parallelism
#'   # =========================================================================
#'   # We want to tune 'alpha' and the RBF 'gamma' parameter.
#'   
#'   grid_rbf <- list(
#'     kernel = "rbf",
#'     alpha  = c(0.01, 1, 10),
#'     gamma  = c(0.001, 0.01, 0.1)
#'   )
#'   
#'   res_rbf <- tune_fastsvm(
#'     data = df, 
#'     time_col = "tempo", delta_col = "cens",
#'     param_grid = grid_rbf,
#'     cv = 3, 
#'     cores = 2, # Passed to Python's joblib
#'     verbose = 1
#'   )
#'   print(res_rbf)
#'
#'   # =========================================================================
#'   # CASE 2: Custom Kernel in R - Uses mirai Parallelism
#'   # =========================================================================
#'   
#'   # 1. Define a Kernel Factory (e.g., simple Sum-Product kernel)
#'   make_sumprod <- function(bias = 0) { 
#'     force(bias)
#'     function(x, z) {
#'       prod(as.numeric(x) * as.numeric(z)) + bias
#'     }
#'   }
#'   
#'   # 2. Create variants to tune the 'bias' parameter
#'   #    This creates a list of functions with metadata
#'   k_variants <- create_kernel_variants(make_sumprod, bias = c(0, 1, 5))
#'   
#'   # 3. Define grid
#'   grid_custom <- list(
#'     kernel = k_variants,   # The kernel function itself is a parameter
#'     alpha  = c(0.1, 1)     # Standard regularization
#'   )
#'   
#'   # 4. Tune (Automatically detects custom kernel -> switches to mirai)
#'   res_custom <- tune_fastsvm(
#'     data = df, 
#'     time_col = "tempo", delta_col = "cens",
#'     param_grid = grid_custom,
#'     cv = 3, 
#'     cores = 2, # Starts 2 background R daemons
#'     verbose = 1
#'   )
#'   print(res_custom)
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
  cores      = parallel::detectCores(),
  verbose    = 0L,
  refit      = TRUE,
  ...
) {
  # --- Validation ---
  if (!is.data.frame(data)) stop("Argument `data` must be a data.frame.")
  if (!time_col %in% names(data)) stop(sprintf("Column '%s' not found.", time_col))
  if (!delta_col %in% names(data)) stop(sprintf("Column '%s' not found.", delta_col))

  # --- UI Setup ---
  has_cli <- requireNamespace("cli", quietly = TRUE)
  has_emo <- requireNamespace("emo", quietly = TRUE)
  ji <- function(x, f="") if(has_emo) emo::ji(x) else f
  
  # --- Imports & Setup ---
  sklearn_sel <- reticulate::import("sklearn.model_selection", delay_load = TRUE)
  sksvm_mod   <- reticulate::import("sksurv.svm", delay_load = TRUE)
  sksurv_util <- reticulate::import("sksurv.util", delay_load = TRUE)
  try(reticulate::py_run_string("import warnings; warnings.simplefilter('ignore')"), silent = TRUE)
  
  # --- Data Prep ---
  x_cols <- setdiff(names(data), c(time_col, delta_col))
  X_mat  <- as.matrix(data[, x_cols, drop = FALSE])
  
  # Extract vectors for safe passing to workers
  time_vec  <- as.numeric(data[[time_col]])
  event_vec <- as.logical(data[[delta_col]])
  
  y_surv <- sksurv_util$Surv$from_arrays(
    event = event_vec, 
    time  = time_vec
  )

  # --- Detect Custom Kernel ---
  is_custom_kernel <- function(grid) {
    check_val <- function(x) is.function(x) || inherits(x, "fastsvm_custom_kernel")
    if (is.list(grid) && is.null(names(grid))) { # List of lists
      any(vapply(grid, function(g) any(vapply(g, function(v) any(vapply(v, check_val, logical(1))), logical(1))), logical(1)))
    } else {
      any(vapply(grid, function(v) any(vapply(v, check_val, logical(1))), logical(1)))
    }
  }
  
  has_custom <- is_custom_kernel(param_grid)
  use_r_parallel <- has_custom && (cores > 1L) && requireNamespace("mirai", quietly = TRUE)

  # ============================================================================
  # BRANCH 1: R Parallelism (Mirai) - For Custom Kernels
  # ============================================================================
  if (use_r_parallel) {
    if (verbose > 0 && has_cli) {
      cli::cli_alert_info("Custom kernel detected. Using R parallelism (mirai) on {cores} cores.")
    }
    
    candidates <- .flatten_sklearn_grid(param_grid)
    
    # Setup Workers
    mirai::daemons(cores, dispatcher = TRUE)
    on.exit(mirai::daemons(0), add = TRUE)
    
    # Worker Function
    run_candidate <- function(params, X, time, event, cv, fixed_args) {
      if (!requireNamespace("reticulate", quietly = TRUE)) return(-Inf)
      
      # Attempt to load package, otherwise safe imports
      try(requireNamespace("FastSurvivalSVM", quietly = TRUE), silent = TRUE)
      
      # Python Setup inside worker
      if (!reticulate::py_available(initialize = TRUE)) reticulate::py_config()
      # Suppress warnings in worker
      try(reticulate::py_run_string("import warnings; warnings.simplefilter('ignore')"), silent = TRUE)
      
      sk_ms  <- reticulate::import("sklearn.model_selection")
      sksvm  <- reticulate::import("sksurv.svm")
      skutil <- reticulate::import("sksurv.util")
      
      # Reconstruct Y inside worker
      y_inner <- skutil$Surv$from_arrays(event = event, time = time)
      
      # Extract kernel function
      k_func <- params$kernel
      
      # Prepare args
      fit_args <- c(fixed_args, params)
      fit_args$kernel <- NULL 
      
      if (is.function(k_func)) {
        py_k <- function(x, z) k_func(as.numeric(x), as.numeric(z))
        fit_args$kernel <- py_k
      } else {
        fit_args$kernel <- k_func
      }
      
      est <- do.call(sksvm$FastKernelSurvivalSVM, fit_args)
      
      # Run CV (Serial inside worker to avoid pickle errors)
      scores <- sk_ms$cross_val_score(est, X, y_inner, cv = as.integer(cv), n_jobs = 1L)
      mean(scores)
    }
    
    # Dispatch
    if (verbose > 0 && has_cli) cli::cli_progress_step("Evaluating {length(candidates)} candidates...", spinner = TRUE)
    
    fixed_args <- list(...)
    promises <- purrr::map(candidates, function(cand) {
      mirai::mirai(
        run_candidate(cand, X_mat, time_vec, event_vec, cv, fixed_args),
        run_candidate = run_candidate, cand = cand, 
        X_mat = X_mat, time_vec = time_vec, event_vec = event_vec,
        cv = cv, fixed_args = fixed_args
      )
    })
    
    # Collect Safely
    scores <- purrr::map_dbl(promises, function(p) {
      out <- mirai::call_mirai(p)$data
      # Error handling
      if (inherits(out, "miraiError") || inherits(out, "error")) return(-Inf)
      if (!is.numeric(out)) return(-Inf)
      out
    })
    
    # Find Best
    best_idx    <- which.max(scores)
    best_score  <- scores[best_idx]
    best_params <- candidates[[best_idx]]
    
    # Refit (Serial in main process)
    best_estimator <- NULL
    if (refit && is.finite(best_score)) {
      final_args <- c(fixed_args, best_params)
      if (is.function(final_args$kernel)) {
        k_r <- final_args$kernel
        final_args$kernel <- function(x, z) k_r(as.numeric(x), as.numeric(z))
      }
      best_estimator <- do.call(sksvm_mod$FastKernelSurvivalSVM, final_args)
      best_estimator$fit(X_mat, y_surv)
    }
    
    # Result Table
    cv_results <- data.frame(
      mean_test_score = scores,
      rank_test_score = rank(-scores)
    )
    
    if (verbose > 0 && has_cli) {
      cli::cli_progress_done()
      cli::cli_alert_success("Tuning complete. Best C-index: {.val {round(best_score, 4)}}")
    }
    
    return(structure(
      list(
        grid_search_obj = NULL, 
        best_estimator  = best_estimator, 
        best_params     = .simplify_params(best_params),
        best_score      = best_score,
        cv_results      = cv_results,
        x_cols = x_cols, time_col = time_col, delta_col = delta_col
      ),
      class = "fastsvm_grid"
    ))
  }

  # ============================================================================
  # BRANCH 2: Python Parallelism (Joblib) - For Standard Kernels or Serial
  # ============================================================================
  
  if (verbose > 0 && has_cli) {
    cli::cli_h2(paste(ji("gear"), "Starting Grid Search Tuning"))
    cli::cli_alert_info("CV Folds: {cv} | Cores: {cores}")
  }

  # Sanitize Grid
  sanitize_single <- function(block) {
    lapply(block, function(val) {
      if (is.function(val)) return(list(val))
      if (is.list(val) && length(val) > 0 && (is.function(val[[1]]) || inherits(val[[1]], "fastsvm_custom_kernel"))) return(unname(val))
      if (!is.list(val) && !is.vector(val)) return(list(val))
      if (length(val) == 1 && !is.list(val)) return(list(val))
      as.list(val)
    })
  }
  is_composite <- is.list(param_grid) && is.null(names(param_grid))
  clean_grid <- if (is_composite) lapply(param_grid, sanitize_single) else sanitize_single(param_grid)

  # Estimator
  fixed_args <- list(...)
  for (p in c("max_iter", "degree", "random_state", "verbose")) {
    if (!is.null(fixed_args[[p]])) fixed_args[[p]] <- as.integer(fixed_args[[p]])
  }
  estimator <- do.call(sksvm_mod$FastKernelSurvivalSVM, fixed_args)

  # If custom kernel present but cores=1, enforce n_jobs=1
  py_n_jobs <- if (has_custom) 1L else as.integer(cores)

  gs_instance <- sklearn_sel$GridSearchCV(
    estimator = estimator, param_grid = clean_grid, cv = as.integer(cv),
    n_jobs = py_n_jobs,
    verbose = as.integer(0),
    refit = refit
  )

  if (verbose > 0 && has_cli) cli::cli_progress_step("Fitting models...", spinner = TRUE)
  
  tryCatch({
    gs_instance$fit(X_mat, y_surv)
  }, error = function(e) {
    if (grepl("pickle", e$message, ignore.case = TRUE)) {
      stop("Pickle Error: Ensure 'cores = 1' is used for custom R kernels if mirai is unavailable.", call. = FALSE)
    } else {
      stop(e)
    }
  })

  # Result Reconstruction
  best_idx   <- tryCatch(as.integer(gs_instance$best_index_) + 1L, error = function(e) NULL)
  best_score <- tryCatch(as.numeric(gs_instance$best_score_), error = function(e) NA)
  
  r_candidates <- .flatten_sklearn_grid(clean_grid)
  best_params <- if (!is.null(best_idx) && best_idx <= length(r_candidates)) r_candidates[[best_idx]] else list()
  best_params <- .simplify_params(best_params)

  py_res <- tryCatch(reticulate::py_to_r(gs_instance$cv_results_), error = function(e) NULL)
  
  cv_results <- NULL
  if (!is.null(py_res)) {
    # Attempt to rebuild readable table
    params_df <- tryCatch({
      df <- do.call(rbind, lapply(r_candidates, function(row) {
        lapply(row, function(val) {
          if (inherits(val, "fastsvm_custom_kernel")) return(attr(val, "kernel_name"))
          if (is.function(val)) return("<function>")
          val
        })
      }))
      as.data.frame(df, stringsAsFactors = FALSE)
    }, error = function(e) NULL)
    
    scores_df <- data.frame(
      mean_test_score = as.numeric(py_res$mean_test_score),
      rank_test_score = as.integer(py_res$rank_test_score)
    )
    
    cv_results <- if(!is.null(params_df)) cbind(scores_df, params_df) else scores_df
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
#' This function allows mixing **native scikit-learn kernels** (string based)
#' and **custom R kernels** (function based) in a single tuning session.
#' 
#' @section Parallelization Details:
#' The \code{cores} parameter is passed down to the individual tuning function.
#' \itemize{
#'   \item For native kernels (strings), parallelism is handled via Scikit-Learn (efficient multithreading).
#'   \item For custom kernels (R functions), parallelism is managed via \pkg{mirai} (R multiprocessing).
#' }
#'
#' @param data Training data frame.
#' @param time_col Time column name.
#' @param delta_col Event column name.
#' @param kernel_mix Named list of base kernel configurations (e.g. \code{list(rbf=list(kernel="rbf"))}).
#' @param param_grids Named list of parameter grids corresponding to \code{kernel_mix}.
#' @param cv Number of folds (default 5).
#' @param cores Number of parallel cores (default: \code{parallel::detectCores()}).
#' @param verbose Verbosity level (0 or 1).
#' @param ... Additional fixed parameters passed to all estimators.
#'
#' @return An object of class \code{"random_machines_tune"}.
#'
#' @examples
#' \dontrun{
#' if (reticulate::py_module_available("sksurv") && requireNamespace("mirai", quietly=TRUE)) {
#'   library(FastSurvivalSVM)
#'   
#'   set.seed(99)
#'   df <- data_generation(n = 200, prop_cen = 0.25)
#'
#'   # =========================================================================
#'   # Setup: Hybrid Tuning (Native + Custom Kernels)
#'   # =========================================================================
#'   
#'   # 1. Prepare Custom Kernel Variants (e.g. Wavelet)
#'   make_wavelet <- function(A = 1) {
#'     force(A)
#'     function(x, z) {
#'       u <- (as.numeric(x) - as.numeric(z)) / A
#'       prod(cos(1.75 * u) * exp(-0.5 * u^2))
#'     }
#'   }
#'   wav_variants <- create_kernel_variants(make_wavelet, A = c(0.5, 2.0))
#'
#'   # 2. Define Base Configurations (The "Mix")
#'   #    Notice we can mix "rbf" (string) and custom functions.
#'   mix <- list(
#'     # A. Native Scikit-learn Kernel
#'     my_rbf = list(
#'       kernel = "rbf",
#'       rank_ratio = 0.5  # Fixed param for this kernel
#'     ),
#'     
#'     # B. Native Linear Kernel
#'     my_linear = list(
#'       kernel = "linear",
#'       rank_ratio = 0.0
#'     ),
#'     
#'     # C. Custom R Kernel
#'     my_wavelet = list(
#'       # We don't set 'kernel' here because we will tune it in the grid
#'       rank_ratio = 1.0
#'     )
#'   )
#'
#'   # 3. Define Grids for each member of the Mix
#'   #    The names must match the 'mix' list.
#'   grids <- list(
#'     # Tune alpha and gamma for RBF
#'     my_rbf = list(
#'       alpha = c(0.1, 10),
#'       gamma = c(0.01, 0.1)
#'     ),
#'     
#'     # Tune only alpha for Linear
#'     my_linear = list(
#'       alpha = c(0.01, 1)
#'     ),
#'     
#'     # Tune the kernel function itself (A=0.5 vs A=2.0) and alpha
#'     my_wavelet = list(
#'       kernel = wav_variants,
#'       alpha  = c(0.1, 1)
#'     )
#'   )
#'
#'   # 4. Run Hybrid Tuning
#'   #    - 'my_rbf' and 'my_linear' will use Python parallelism (fast)
#'   #    - 'my_wavelet' will use mirai parallelism (robust)
#'   tune_results <- tune_random_machines(
#'     data = df,
#'     time_col = "tempo", delta_col = "cens",
#'     kernel_mix = mix,
#'     param_grids = grids,
#'     cv = 3,
#'     cores = 2,
#'     verbose = 1
#'   )
#'   
#'   print(tune_results)
#' }
#' }
#' @export
tune_random_machines <- function(
  data,
  time_col  = "t",
  delta_col = "delta",
  kernel_mix,
  param_grids,
  cv        = 5L,
  cores     = parallel::detectCores(),
  verbose   = 0L,
  ...
) {
  if (!is.data.frame(data)) stop("Argument `data` must be a data.frame.")
  common_names <- intersect(names(kernel_mix), names(param_grids))
  
  if (verbose > 0 && requireNamespace("cli", quietly=TRUE)) {
    cli::cli_h1("Multi-Kernel Tuning Session")
    cli::cli_alert_info("Optimizing {length(common_names)} kernel configurations on {cores} cores.")
  }
  
  results <- list()
  
  for (kname in common_names) {
    base_def <- kernel_mix[[kname]]
    grid_def <- param_grids[[kname]]
    
    if (verbose > 0 && requireNamespace("cli", quietly=TRUE)) {
      cli::cli_h2(paste("Tuning:", kname))
    }
    
    fixed_args <- base_def
    # Remove from fixed args anything that varies in the grid
    for (p in names(grid_def)) fixed_args[[p]] <- NULL
    
    res <- tryCatch({
      do.call(tune_fastsvm, c(
        list(
          data       = data,
          time_col   = time_col,
          delta_col  = delta_col,
          param_grid = grid_def,
          cv         = cv,
          cores      = cores, # Pass cores down
          verbose    = 0,     # Silence internal
          refit      = TRUE
        ),
        fixed_args, 
        list(...)
      ))
    }, error = function(e) {
      if (requireNamespace("cli", quietly=TRUE)) cli::cli_alert_danger("Failed: {e$message}")
      NULL
    })
    
    if (verbose > 0 && !is.null(res) && requireNamespace("cli", quietly=TRUE)) {
      cli::cli_alert_success("Best C-index: {.val {round(res$best_score, 4)}}")
    }
    results[[kname]] <- res
  }
  
  structure(results, class = "random_machines_tune")
}

# -------------------------------------------------------------------
# S3 Print Methods
# -------------------------------------------------------------------

#' Print method for Random Machines tuning results
#' 
#' @param x An object of class \code{"random_machines_tune"}.
#' @param ... Additional arguments passed to methods.
#' 
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
    print(x) 
  }
  invisible(x)
}

#' Print method for single tuning result
#' 
#' @param x An object of class \code{"fastsvm_grid"}.
#' @param ... Additional arguments passed to methods.
#' 
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
      val_print <- if (inherits(val, "fastsvm_custom_kernel")) attr(val, "kernel_name") else as.character(val)
      cli::cli_li("{.emph {nm}}: {val_print}")
    }
    cli::cli_end()
  } else {
    cat(sprintf("Best C-Index: %.4f\n", x$best_score))
    print(x$best_params)
  }
  invisible(x)
}
