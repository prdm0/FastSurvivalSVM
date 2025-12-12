# R/tune.R

# ==============================================================================
#  PART 1: Custom Kernel Helpers & Metadata
# ==============================================================================

#' Create a Kernel Grid or Single Instance (Smart Wrapper)
#'
#' This helper simplifies the definition of custom kernels. It serves two purposes:
#' \enumerate{
#'   \item **Tuning:** If you provide vectors of parameters, it returns a list of
#'         kernel variants to be used in \code{\link{tune_random_machines}} or
#'         \code{\link{tune_fastsvm}}.
#'   \item **Fixed Instance:** If you provide single values for parameters, it
#'         returns the single kernel function directly, ready for \code{\link{random_machines}}.
#' }
#'
#' This abstracts away the complexity of the "function factory" pattern (closures)
#' and eliminates the need to manually extract elements from a list when creating
#' a single kernel.
#'
#' @param func A bare R function. It must accept at least two arguments named
#'   \code{x} and \code{z} (representing the data vectors), followed by any
#'   hyperparameters you wish to tune.
#' @param ... Named arguments defining the hyperparameters.
#'   \itemize{
#'     \item **Vectors:** Will be expanded into a grid (returns a list).
#'     \item **Scalars:** Will create a single instance (returns the function).
#'   }
#'
#' @return
#' \itemize{
#'   \item If the expansion results in multiple variants: A named list of kernel functions.
#'   \item If the expansion results in a single variant: The kernel function itself.
#' }
#'
#' @examples
#' # 1. Define the kernel logic
#' my_wavelet <- function(x, z, A) {
#'   u <- (as.numeric(x) - as.numeric(z)) / A
#'   prod(cos(1.75 * u) * exp(-0.5 * u^2))
#' }
#'
#' # 2. For Tuning: Create a list of variants (vector input)
#' #    Returns a list of 3 functions
#' grid <- grid_kernel(my_wavelet, A = c(0.5, 1, 2))
#' class(grid) # "list"
#'
#' # 3. For Fixed Use: Create a single instance (scalar input)
#' #    Returns the function directly (Smart Return, no [[1]] needed)
#' k_one <- grid_kernel(my_wavelet, A = 1)
#' class(k_one) # "fastsvm_custom_kernel" (function)
#'
#' @export
grid_kernel <- function(func, ...) {
  # 1. Capture parameters
  args_list <- list(...)

  # 2. Basic validation
  if (!is.function(func)) stop("'func' must be a function.")
  func_args <- names(formals(func))
  if (length(intersect(c("x", "z"), func_args)) < 2) {
    stop("Kernel function must accept at least arguments 'x' and 'z'.")
  }

  # 3. Internal Factory
  internal_factory <- function(...) {
    # Capture params for this specific instance (e.g., A=1)
    instance_params <- list(...)

    # Construct the closure that FastSurvivalSVM expects: f(x, z)
    k_closure <- function(x, z) {
      # Combine x, z with the instance parameters and call the user's function
      call_args <- c(list(x = x, z = z), instance_params)
      do.call(func, call_args)
    }
    return(k_closure)
  }

  # 4. Delegate to the low-level mechanism (now internal)
  res <- do.call(create_kernel_variants, c(list(kernel_factory = internal_factory), args_list))

  # 5. Smart Return: If single result, unwrap it to improve UX.
  if (length(res) == 1L) {
    return(res[[1]])
  } else {
    return(res)
  }
}

#' Internal helper to create variants of a custom kernel function
#' @keywords internal
create_kernel_variants <- function(kernel_factory, ...) {
  args_grid <- expand.grid(..., stringsAsFactors = FALSE)
  kernel_list <- list()

  for (i in 1:nrow(args_grid)) {
    current_args <- as.list(args_grid[i, , drop = FALSE])
    k_func <- do.call(kernel_factory, current_args)

    # Format clean name (e.g., "a=1_b=2") for display and debugging
    fmt_args <- lapply(current_args, function(x) if(is.numeric(x)) round(x, 4) else x)
    name <- paste(names(fmt_args), fmt_args, sep = "=", collapse = "_")

    # Attach metadata for pretty printing and reticulate conversion
    attr(k_func, "kernel_name") <- name
    attr(k_func, "kernel_args") <- current_args
    class(k_func) <- c("fastsvm_custom_kernel", class(k_func))

    kernel_list[[name]] <- k_func
  }
  return(kernel_list)
}

#' Print method for custom kernels
#' @param x A custom kernel object.
#' @param ... Additional arguments.
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
#' @param x A custom kernel object.
#' @param name The name of the parameter to access.
#' @export
`$.fastsvm_custom_kernel` <- function(x, name) {
  attr(x, "kernel_args")[[name]]
}

# ==============================================================================
#  PART 2: Internal Helpers (Grid Reconstruction)
# ==============================================================================

.simplify_params <- function(params_list) {
  lapply(params_list, function(x) {
    if (is.list(x) && length(x) == 1 && is.null(names(x))) return(x[[1]])
    x
  })
}

.flatten_sklearn_grid <- function(param_grid) {
  if (!is.null(names(param_grid))) param_grid <- list(param_grid)
  candidates <- list()
  for (block in param_grid) {
    keys <- sort(names(block))
    values_list <- block[keys]
    indices_list <- lapply(values_list, function(x) seq_along(x))
    rev_inds <- rev(indices_list)
    grid_inds <- do.call(expand.grid, c(rev_inds, KEEP.OUT.ATTRS = FALSE))
    if (nrow(grid_inds) > 0) {
      grid_inds <- grid_inds[, rev(names(grid_inds)), drop = FALSE]
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
#' Executes a grid search with cross-validation to optimize hyperparameters for a
#' single kernel configuration.
#'
#' @section Parallelization Strategy:
#' The \code{cores} argument controls the parallel backend:
#' \itemize{
#'   \item \strong{Standard Kernels} (strings like "rbf", "linear"): Uses Python's
#'         \code{joblib} (via scikit-learn) for efficient multi-threading.
#'   \item \strong{Custom R Kernels} (created via \code{\link{grid_kernel}}): Uses
#'         the \pkg{mirai} package to distribute candidates across R background processes.
#' }
#'
#' @param data Training data frame.
#' @param time_col Name of the column containing survival times.
#' @param delta_col Name of the column containing the event indicator (1=event, 0=censored).
#' @param param_grid A named list (or list of lists) of parameters to tune.
#'   \itemize{
#'     \item For **Standard Kernels**: Provide vectors of values.
#'           Example: \code{list(kernel="rbf", gamma=c(0.1, 1))}
#'     \item For **Custom Kernels**: Use \code{\link{grid_kernel}} to generate the kernel list.
#'           Example: \code{list(kernel=grid_kernel(my_func, A=1:3), alpha=c(0.1, 1))}
#'   }
#' @param cv Number of cross-validation folds (default 5).
#' @param cores Number of cores to use (defaults to detected cores).
#' @param verbose Verbosity level (0 or 1).
#' @param refit Logical. If \code{TRUE}, refits the model with the best found parameters on the full dataset.
#' @param ... Additional arguments passed directly to the estimator (e.g., \code{rank_ratio}, \code{max_iter}).
#'
#' @return An object of class \code{"fastsvm_grid"} containing the best parameters, score, and full CV results.
#'
#' @examples
#' \dontrun{
#' if (reticulate::py_module_available("sksurv") && requireNamespace("mirai", quietly = TRUE)) {
#'   library(FastSurvivalSVM)
#'   set.seed(42)
#'
#'   # --- 1. Prepare Data ---
#'   # Generating synthetic survival data
#'   df <- data_generation(n = 200, prop_cen = 0.3)
#'
#'   # =========================================================================
#'   # EXAMPLE A: Tuning a Standard Kernel (RBF)
#'   # =========================================================================
#'   # We want to tune 'alpha' (regularization) and 'gamma' (kernel width).
#'   # rank_ratio = 0 implies we are optimizing for Regression (time), not Ranking.
#'
#'   grid_rbf <- list(
#'     kernel = "rbf",
#'     rank_ratio = 0.0,
#'     alpha  = c(0.01, 0.1, 1.0),
#'     gamma  = c(0.01, 0.1, 1.0)
#'   )
#'
#'   # Automatically uses Scikit-learn parallelism (Python threads)
#'   res_rbf <- tune_fastsvm(
#'     data = df,
#'     time_col = "tempo", delta_col = "cens",
#'     param_grid = grid_rbf,
#'     cv = 3,
#'     cores = parallel::detectCores(),
#'     verbose = 1
#'   )
#'   print(res_rbf)
#'
#'   # =========================================================================
#'   # EXAMPLE B: Tuning a Custom R Kernel (Using grid_kernel)
#'   # =========================================================================
#'
#'   # 1. Define the kernel function (Simple Sum-Product + Bias)
#'   my_sumprod <- function(x, z, bias) {
#'      prod(as.numeric(x) * as.numeric(z)) + bias
#'   }
#'
#'   # 2. Define the grid
#'   #    - 'kernel': use grid_kernel() to vary 'bias' (returns a list of functions)
#'   #    - 'alpha': standard SVM parameter
#'   grid_custom <- list(
#'     kernel = grid_kernel(my_sumprod, bias = c(0, 1, 5)),
#'     rank_ratio = 0.0,
#'     alpha  = c(0.1, 1.0)
#'   )
#'
#'   # 3. Tune (Automatically switches to mirai for R parallelism)
#'   res_custom <- tune_fastsvm(
#'     data = df,
#'     time_col = "tempo", delta_col = "cens",
#'     param_grid = grid_custom,
#'     cv = 3,
#'     cores = parallel::detectCores(),
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
  # --- UI Setup ---
  has_cli <- requireNamespace("cli", quietly = TRUE)

  # --- Imports & Setup ---
  try(reticulate::py_run_string("import warnings; warnings.simplefilter('ignore')"), silent = TRUE)
  sksvm_mod   <- reticulate::import("sksurv.svm", delay_load = TRUE)
  sksurv_util <- reticulate::import("sksurv.util", delay_load = TRUE)

  # --- Data Prep ---
  x_cols <- setdiff(names(data), c(time_col, delta_col))
  X_mat  <- as.matrix(data[, x_cols, drop = FALSE])
  time_vec  <- as.numeric(data[[time_col]])
  event_vec <- as.logical(data[[delta_col]])
  y_surv <- sksurv_util$Surv$from_arrays(event = event_vec, time = time_vec)

  # --- Detect Custom Kernel ---
  is_custom_kernel <- function(grid) {
    check_val <- function(x) is.function(x) || inherits(x, "fastsvm_custom_kernel")
    if (is.list(grid) && is.null(names(grid))) {
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
    mirai::daemons(cores, dispatcher = TRUE)
    on.exit(mirai::daemons(0), add = TRUE)

    run_candidate <- function(params, X, time, event, cv, fixed_args) {
      if (!requireNamespace("reticulate", quietly = TRUE)) return(-Inf)
      try(requireNamespace("FastSurvivalSVM", quietly = TRUE), silent = TRUE)
      if (!reticulate::py_available(initialize = TRUE)) reticulate::py_config()
      try(reticulate::py_run_string("import warnings; warnings.simplefilter('ignore')"), silent = TRUE)

      sk_ms  <- reticulate::import("sklearn.model_selection")
      sksvm  <- reticulate::import("sksurv.svm")
      skutil <- reticulate::import("sksurv.util")

      y_inner <- skutil$Surv$from_arrays(event = event, time = time)
      k_func <- params$kernel
      fit_args <- c(fixed_args, params)
      fit_args$kernel <- NULL

      if (is.function(k_func)) {
        py_k <- function(x, z) k_func(as.numeric(x), as.numeric(z))
        fit_args$kernel <- py_k
      } else {
        fit_args$kernel <- k_func
      }

      est <- do.call(sksvm$FastKernelSurvivalSVM, fit_args)
      scores <- sk_ms$cross_val_score(est, X, y_inner, cv = as.integer(cv), n_jobs = 1L)
      mean(scores)
    }

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

    scores <- purrr::map_dbl(promises, function(p) {
      out <- mirai::call_mirai(p)$data
      if (inherits(out, "miraiError") || inherits(out, "error") || !is.numeric(out)) return(-Inf)
      out
    })

    best_idx    <- which.max(scores)
    best_score  <- scores[best_idx]
    best_params <- candidates[[best_idx]]

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

    cv_results <- data.frame(mean_test_score = scores, rank_test_score = rank(-scores))

    if (verbose > 0 && has_cli) {
      cli::cli_progress_done()
      cli::cli_alert_success("Tuning complete. Best C-index: {.val {round(best_score, 4)}}")
    }

    return(structure(
      list(best_estimator = best_estimator, best_params = .simplify_params(best_params),
           best_score = best_score, cv_results = cv_results,
           x_cols = x_cols, time_col = time_col, delta_col = delta_col),
      class = "fastsvm_grid"
    ))
  }

  # ============================================================================
  # BRANCH 2: Python Parallelism (Standard Kernels)
  # ============================================================================
  if (verbose > 0 && has_cli) {
    cli::cli_h2("Starting Grid Search Tuning (Standard Kernel)")
    cli::cli_alert_info("CV: {cv} | Cores: {cores}")
  }

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

  fixed_args <- list(...)
  for (p in c("max_iter", "degree", "random_state", "verbose")) {
    if (!is.null(fixed_args[[p]])) fixed_args[[p]] <- as.integer(fixed_args[[p]])
  }
  estimator <- do.call(sksvm_mod$FastKernelSurvivalSVM, fixed_args)
  py_n_jobs <- if (has_custom) 1L else as.integer(cores)

  sklearn_sel <- reticulate::import("sklearn.model_selection", delay_load = TRUE)
  gs_instance <- sklearn_sel$GridSearchCV(
    estimator = estimator, param_grid = clean_grid, cv = as.integer(cv),
    n_jobs = py_n_jobs, verbose = as.integer(0), refit = refit
  )

  if (verbose > 0 && has_cli) cli::cli_progress_step("Fitting models...", spinner = TRUE)
  tryCatch({ gs_instance$fit(X_mat, y_surv) }, error = function(e) stop(e))

  best_idx   <- tryCatch(as.integer(gs_instance$best_index_) + 1L, error = function(e) NULL)
  best_score <- tryCatch(as.numeric(gs_instance$best_score_), error = function(e) NA)
  r_candidates <- .flatten_sklearn_grid(clean_grid)
  best_params <- if (!is.null(best_idx)) r_candidates[[best_idx]] else list()
  best_params <- .simplify_params(best_params)

  py_res <- tryCatch(reticulate::py_to_r(gs_instance$cv_results_), error = function(e) NULL)
  cv_results <- if (!is.null(py_res)) {
    scores_df <- data.frame(mean_test_score = as.numeric(py_res$mean_test_score),
                            rank_test_score = as.integer(py_res$rank_test_score))
    scores_df
  } else NULL

  if (verbose > 0 && has_cli) {
    cli::cli_progress_done()
    cli::cli_alert_success("Tuning complete. Best C-index: {.val {round(best_score, 4)}}")
  }

  structure(
    list(grid_search_obj = gs_instance, best_estimator = gs_instance$best_estimator_,
         best_params = best_params, best_score = best_score, cv_results = cv_results,
         x_cols = x_cols, time_col = time_col, delta_col = delta_col),
    class = "fastsvm_grid"
  )
}

#' Multi-Kernel Tuning for Random Machines
#'
#' Orchestrates hyperparameter tuning for multiple kernels simultaneously.
#'
#' @param data Training data frame.
#' @param time_col Time column name.
#' @param delta_col Event column name.
#' @param kernel_mix A named list of base configurations.
#' @param param_grids A named list of grids.
#' @param cv Number of folds (default 5).
#' @param cores Number of parallel cores (default: \code{parallel::detectCores()}).
#' @param verbose Verbosity level (0 or 1).
#' @param ... Additional fixed parameters.
#'
#' @return An object of class \code{"random_machines_tune"}.
#'
#' @examples
#' \dontrun{
#' if (reticulate::py_module_available("sksurv") && requireNamespace("mirai")) {
#'   library(FastSurvivalSVM)
#'   set.seed(99)
#'
#'   # --- 1. Prepare Data ---
#'   df <- data_generation(n = 300, prop_cen = 0.25)
#'
#'   # =========================================================================
#'   # 2. Define Custom Kernel Functions
#'   # =========================================================================
#'
#'   # Wavelet Kernel (Custom 1)
#'   my_wavelet <- function(x, z, A) {
#'     u <- (as.numeric(x) - as.numeric(z)) / A
#'     prod(cos(1.75 * u) * exp(-0.5 * u^2))
#'   }
#'
#'   # Polynomial Kernel (Custom 2)
#'   my_poly <- function(x, z, degree, coef0) {
#'     val <- sum(as.numeric(x) * as.numeric(z)) + coef0
#'     val ^ degree
#'   }
#'
#'   # =========================================================================
#'   # 3. Define Kernel Structure (Regression Mode: rank_ratio = 0)
#'   # =========================================================================
#'   kernel_mix <- list(
#'     linear_std = list(kernel = "linear", rank_ratio = 0.0),
#'     rbf_std    = list(kernel = "rbf",    rank_ratio = 0.0),
#'     wavelet_ok = list(rank_ratio = 0.0),
#'     poly_ok    = list(rank_ratio = 0.0)
#'   )
#'
#'   # =========================================================================
#'   # 4. Define Grids (4 Kernels x 4 Values per parameter)
#'   # =========================================================================
#'
#'   param_grids <- list(
#'     # 1. Linear (Native): 4 alpha values
#'     linear_std = list(
#'       alpha = c(0.01, 0.1, 1.0, 10.0)
#'     ),
#'
#'     # 2. RBF (Native): 4 alpha values x 4 gamma values
#'     rbf_std = list(
#'       alpha = c(0.01, 0.1, 1.0, 10.0),
#'       gamma = c(0.001, 0.01, 0.1, 1.0)
#'     ),
#'
#'     # 3. Wavelet (Custom): 4 variants (A) x 4 alphas
#'     wavelet_ok = list(
#'       kernel = grid_kernel(my_wavelet, A = c(0.5, 1.0, 1.5, 2.0)),
#'       alpha  = c(0.01, 0.1, 1.0, 10.0)
#'     ),
#'
#'     # 4. Polynomial (Custom): 4 variants (degree) x 4 alphas
#'     # Note: We fix coef0=1 to reduce grid explosion, but degree varies 4 times
#'     poly_ok = list(
#'       kernel = grid_kernel(my_poly, degree = c(1, 2, 3, 4), coef0 = 1),
#'       alpha  = c(0.01, 0.1, 1.0, 10.0)
#'     )
#'   )
#'
#'   # =========================================================================
#'   # 5. Run Tuning (Using ALL cores)
#'   # =========================================================================
#'   tune_res <- tune_random_machines(
#'     data        = df,
#'     time_col    = "tempo",
#'     delta_col   = "cens",
#'     kernel_mix  = kernel_mix,
#'     param_grids = param_grids,
#'     cv          = 3,
#'     cores       = parallel::detectCores(),
#'     verbose     = 1
#'   )
#'
#'   print(tune_res)
#'
#'   # 6. Bridge to Training
#'   final_kernels <- as_kernels(tune_res, kernel_mix)
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
    for (p in names(grid_def)) fixed_args[[p]] <- NULL

    res <- tryCatch({
      do.call(tune_fastsvm, c(
        list(data = data, time_col = time_col, delta_col = delta_col,
             param_grid = grid_def, cv = cv, cores = cores, verbose = 0, refit = TRUE),
        fixed_args, list(...)
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

#' Print method for Random Machines tuning results
#' @param x Tuning result object.
#' @param ... Additional arguments.
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
#' @param x Grid search result object.
#' @param ... Additional arguments.
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

#' Prepare Tuned Kernels for Random Machines
#' @param tune_results The results from \code{tune_random_machines}.
#' @param kernel_mix The original base configuration list.
#' @export
as_kernels <- function(tune_results, kernel_mix) {
  if (!inherits(tune_results, "random_machines_tune")) {
    stop("Argument 'tune_results' must be of class 'random_machines_tune'.")
  }
  tune_names <- names(tune_results)
  mix_names  <- names(kernel_mix)

  merge_one <- function(kname) {
    base_conf <- kernel_mix[[kname]]
    if (is.null(base_conf)) base_conf <- list()
    if (is.null(tune_results[[kname]])) return(base_conf)
    tuned_params <- tune_results[[kname]]$best_params
    utils::modifyList(base_conf, tuned_params)
  }

  final_list <- lapply(tune_names, merge_one)
  names(final_list) <- tune_names
  return(final_list)
}