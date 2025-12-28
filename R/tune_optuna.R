# R/tune_optuna.R

# ==============================================================================
#  Optuna Tuning Infrastructure for FastSurvivalSVM
# ==============================================================================

#' Define Float Parameter for Optuna Search Space
#'
#' Helper function to define a continuous floating point hyperparameter range
#' for Bayesian optimization.
#'
#' @param low Numeric. The lower bound of the range.
#' @param high Numeric. The upper bound of the range.
#' @param log Logical. If \code{TRUE}, the value is sampled from the range in the log domain.
#'   Useful for parameters like learning rates or regularization (alpha, gamma).
#' @param step Numeric or NULL. The discretization step. If provided, the value will be rounded
#'   to the nearest multiple of step.
#'
#' @return An object of class \code{"opt_param_def"} defining the search strategy.
#' @export
#'
#' @examples
#' # Define a log-scale search for alpha between 0.001 and 100
#' space_alpha <- opt_float(0.001, 100, log = TRUE)
opt_float <- function(low, high, log = FALSE, step = NULL) {
  structure(list(type = "float", low = low, high = high, log = log, step = step),
            class = "opt_param_def")
}

#' Define Integer Parameter for Optuna Search Space
#'
#' Helper function to define an integer hyperparameter range for Bayesian optimization.
#'
#' @param low Integer. The lower bound of the range.
#' @param high Integer. The upper bound of the range.
#' @param log Logical. If \code{TRUE}, the value is sampled from the range in the log domain.
#' @param step Integer. The discretization step (default 1).
#'
#' @return An object of class \code{"opt_param_def"}.
#' @export
#'
#' @examples
#' # Define search for polynomial degree between 2 and 5
#' space_degree <- opt_int(2, 5)
opt_int <- function(low, high, log = FALSE, step = 1L) {
  structure(list(type = "int", low = as.integer(low), high = as.integer(high), log = log, step = step),
            class = "opt_param_def")
}

#' Define Categorical Parameter for Optuna Search Space
#'
#' Helper function to define a categorical choice set for Bayesian optimization.
#'
#' @param choices A vector of values (numeric, character, or logical) to choose from.
#'
#' @return An object of class \code{"opt_param_def"}.
#' @export
#'
#' @examples
#' # Choose between different optimizers
#' space_opt <- opt_cat(c("avltree", "rbtree"))
opt_cat <- function(choices) {
  structure(list(type = "categorical", choices = choices),
            class = "opt_param_def")
}

# ------------------------------------------------------------------------------
#  Worker Function (Runs inside parallel R processes)
# ------------------------------------------------------------------------------

#' Internal worker function for Optuna Distributed Optimization
#'
#' This function is executed inside parallel R workers initiated by
#' \code{tune_fastsvm_optuna}. It connects to a shared SQLite database,
#' requests trials from the Optuna study, runs the model evaluation (CV),
#' and reports the objective value back to the database.
#'
#' @keywords internal
.optuna_worker <- function(
    process_id,
    n_trials_per_worker,
    storage_url,
    study_name,
    data,
    time_col,
    delta_col,
    search_space,
    base_args,
    cv,
    seed
) {
  # 1. Environment Setup
  if (!requireNamespace("reticulate", quietly = TRUE)) return(NULL)
  
  # Ensure Python is loaded
  if (!reticulate::py_available(initialize = TRUE)) reticulate::py_config()
  
  # Import Python Libraries locally within the worker
  optuna <- reticulate::import("optuna", delay_load = FALSE)
  sksvm  <- reticulate::import("sksurv.svm", delay_load = FALSE)
  skms   <- reticulate::import("sklearn.model_selection", delay_load = FALSE)
  skutil <- reticulate::import("sksurv.util", delay_load = FALSE)
  
  # Suppress warnings for cleaner output
  reticulate::py_run_string("import warnings; warnings.simplefilter('ignore')")
  optuna$logging$set_verbosity(optuna$logging$WARNING)
  
  # 2. Data Preparation (Once per worker to save overhead)
  #    We convert R dataframes to Python matrices/arrays immediately.
  x_cols <- setdiff(names(data), c(time_col, delta_col))
  X_mat  <- as.matrix(data[, x_cols, drop = FALSE])
  
  y_surv <- skutil$Surv$from_arrays(
    event = as.logical(data[[delta_col]]),
    time  = as.numeric(data[[time_col]])
  )
  
  # 3. Define the Objective Function
  #    This function runs inside the Optuna loop
  objective_fun <- function(trial) {
    
    # A. Parse the search space from R definitions to Optuna calls
    current_params <- list()
    for (p_name in names(search_space)) {
      def <- search_space[[p_name]]
      
      val <- switch(
        def$type,
        "float" = trial$suggest_float(p_name, def$low, def$high, log = def$log, step = def$step),
        "int"   = trial$suggest_int(p_name, def$low, def$high, log = def$log, step = def$step),
        "categorical" = trial$suggest_categorical(p_name, def$choices)
      )
      current_params[[p_name]] <- val
    }
    
    # B. Merge with base arguments (kernel type, fixed params)
    #    current_params overrides base_args if duplicates exist
    full_args <- utils::modifyList(base_args, current_params)
    
    # C. Handle Custom Kernel Logic
    #    If 'kernel' is a function, we must wrap it for Python to include
    #    the CURRENT hyperparameters (e.g., 'A' for wavelet) if they were tuned.
    if (is.function(full_args$kernel)) {
      r_kernel_func <- full_args$kernel
      
      # Extract parameters that the kernel function expects (excluding x, z)
      k_formals <- names(formals(r_kernel_func))
      k_param_names <- setdiff(k_formals, c("x", "z"))
      
      # Capture the values for these parameters from the current trial
      k_params_vals <- full_args[intersect(names(full_args), k_param_names)]
      
      # FIX: Force numeric conversion to avoid Numpy scalar issues in arithmetic
      k_params_vals <- lapply(k_params_vals, as.numeric)
      
      # Create the closure: k(x, z)
      kernel_py <- function(x, z) {
        # Bridge: Numeric vectors
        x_r <- as.numeric(x)
        z_r <- as.numeric(z)
        
        # Call original R function with current trial's params
        do.call(r_kernel_func, c(list(x = x_r, z = z_r), k_params_vals))
      }
      
      full_args$kernel <- kernel_py
    }
    
    # D. Clean arguments for the Python Constructor
    #    FastKernelSurvivalSVM __init__ only accepts specific args.
    valid_keys <- c("kernel", "alpha", "rank_ratio", "fit_intercept", 
                    "max_iter", "tol", "verbose", "optimizer", "random_state")
    
    ctor_args <- full_args[names(full_args) %in% valid_keys]
    
    # E. Train and Evaluate
    tryCatch({
      estimator <- do.call(sksvm$FastKernelSurvivalSVM, ctor_args)
      
      # Parallelism note: We use n_jobs=1 here because the OUTER loop (Optuna)
      # is already distributed across R processes.
      scores <- skms$cross_val_score(
        estimator, X_mat, y_surv, 
        cv = as.integer(cv), 
        n_jobs = 1L 
      )
      
      mean(as.numeric(scores))
    }, error = function(e) {
      return(-1.0)
    })
  }
  
  # 4. Connect to Shared Study and Optimize
  #    Reticulate handles the SQLite connection via the Python library
  study <- optuna$load_study(study_name = study_name, storage = storage_url)
  
  # Run optimization part for this worker
  study$optimize(objective_fun, n_trials = as.integer(n_trials_per_worker))
  
  return(TRUE)
}

# ------------------------------------------------------------------------------
#  Main Tuning Function (Single Kernel)
# ------------------------------------------------------------------------------

#' Tune FastSurvivalSVM using Optuna (Single Kernel Optimization)
#'
#' Performs Bayesian Optimization using Optuna to find the best hyperparameters
#' for a single kernel configuration. The search is parallelized by launching
#' multiple R workers (via \code{parallel::makePSOCKcluster}) that connect to a
#' shared SQLite database to synchronize trials.
#'
#' @param data A \code{data.frame} containing the training data.
#' @param time_col Character string. Name of the time column.
#' @param delta_col Character string. Name of the event column (1=event, 0=censored).
#' @param search_space A named list defining the parameters to tune.
#'   Use helper functions \code{\link{opt_float}}, \code{\link{opt_int}},
#'   or \code{\link{opt_cat}}.
#' @param n_trials Integer. Total number of trials to run (these are distributed
#'   across the \code{cores}).
#' @param cv Integer. Number of cross-validation folds to use for evaluating
#'   each trial (default 5).
#' @param cores Integer. Number of parallel R workers to launch.
#' @param seed Integer or NULL. Random seed for reproducibility (passed to Optuna's sampler).
#' @param verbose Logical. If TRUE, prints start and end messages.
#' @param ... Additional fixed arguments passed to \code{FastKernelSurvivalSVM}
#'   (e.g., \code{kernel="rbf"}, \code{max_iter=1000}).
#'
#' @return An object of class \code{"fastsvm_optuna"} containing:
#'   \item{best_params}{A list of the optimal hyperparameters found.}
#'   \item{best_score}{The best mean C-index achieved.}
#'   \item{n_trials}{The number of trials performed.}
#'   \item{study}{The original Python Optuna study object.}
#'
#' @examples
#' \dontrun{
#' if (reticulate::py_module_available("optuna") && requireNamespace("parallel")) {
#'   library(FastSurvivalSVM)
#'   
#'   # 1. Generate Data
#'   df <- data_generation(n = 200, prop_cen = 0.3)
#'   
#'   # 2. Define Search Space for an RBF Kernel
#'   #    We want to tune 'alpha' and 'gamma'.
#'   space <- list(
#'     alpha = opt_float(0.01, 10, log = TRUE),
#'     gamma = opt_float(0.001, 1, log = TRUE)
#'   )
#'   
#'   # 3. Run Optimization using ALL available cores
#'   res <- tune_fastsvm_optuna(
#'     data         = df,
#'     time_col     = "tempo",
#'     delta_col    = "cens",
#'     search_space = space,
#'     n_trials     = 20,
#'     cv           = 3,
#'     cores        = parallel::detectCores(),
#'     kernel       = "rbf",
#'     rank_ratio   = 0.0 # Fixed parameter
#'   )
#'   
#'   print(res$best_params)
#'   print(res$best_score)
#'
#'   # 4. Train Final Model with Best Parameters
#'   #    Using the optimized hyperparameters to fit the model on full data.
#'
#'   # Combine fixed arguments (kernel, rank_ratio) with tuned ones (alpha, gamma)
#'   final_args <- c(
#'     list(
#'       data       = df,
#'       time_col   = "tempo",
#'       delta_col  = "cens",
#'       kernel     = "rbf",
#'       rank_ratio = 0.0
#'     ),
#'     res$best_params
#'   )
#'   
#'   # Fit using fastsvm()
#'   final_model <- do.call(fastsvm, final_args)
#'   
#'   print(final_model)
#' }
#' }
#' @importFrom parallel makePSOCKcluster stopCluster parLapply clusterExport clusterEvalQ
#' @export
tune_fastsvm_optuna <- function(
    data,
    time_col = "t",
    delta_col = "delta",
    search_space,
    n_trials = 50L,
    cv = 5L,
    cores = parallel::detectCores(),
    seed = NULL,
    verbose = TRUE,
    ...
) {
  if (!requireNamespace("reticulate", quietly = TRUE)) stop("Package 'reticulate' required.")
  if (!requireNamespace("parallel", quietly = TRUE)) stop("Package 'parallel' required.")
  
  # 1. Setup Optuna Storage (The Synchronization Mechanism)
  db_file <- tempfile(pattern = "optuna_", fileext = ".db")
  storage_url <- paste0("sqlite:///", db_file)
  study_name  <- paste0("study_", paste(sample(letters, 8), collapse=""))
  
  # Initialize Optuna in Main Process to create the DB tables
  optuna <- reticulate::import("optuna", delay_load = FALSE)
  optuna$logging$set_verbosity(optuna$logging$WARNING)
  
  sampler <- if (!is.null(seed)) optuna$samplers$TPESampler(seed = as.integer(seed)) else NULL
  
  # Create the study (creates the .db file)
  optuna$create_study(
    study_name = study_name,
    storage = storage_url,
    direction = "maximize",
    load_if_exists = TRUE,
    sampler = sampler
  )
  
  # 2. Distribute Trials
  if (cores > n_trials) cores <- n_trials
  trials_per_worker <- rep(floor(n_trials / cores), cores)
  rem <- n_trials %% cores
  if (rem > 0) trials_per_worker[1:rem] <- trials_per_worker[1:rem] + 1
  
  base_args <- list(...)
  
  # 3. Launch Parallel Workers
  if (verbose) message(sprintf("Starting Optuna: %d trials on %d cores (SQLite sync)...", n_trials, cores))
  
  cl <- parallel::makePSOCKcluster(cores)
  on.exit({
    parallel::stopCluster(cl)
    if (file.exists(db_file)) unlink(db_file)
  }, add = TRUE)
  
  # Export necessary objects/libs to workers
  parallel::clusterEvalQ(cl, {
    library(reticulate)
  })
  
  # Explicitly export worker function and data to avoid scope issues
  parallel::clusterExport(
    cl, 
    varlist = c(".optuna_worker", "data", "time_col", "delta_col", "search_space", "base_args", "cv"), 
    envir = environment()
  )
  
  # Execute
  parallel::parLapply(
    cl,
    seq_len(cores),
    function(i) {
      .optuna_worker(
        process_id = i,
        n_trials_per_worker = trials_per_worker[i],
        storage_url = storage_url,
        study_name = study_name,
        data = data,
        time_col = time_col,
        delta_col = delta_col,
        search_space = search_space,
        base_args = base_args,
        cv = cv,
        seed = if(!is.null(seed)) seed + i else NULL
      )
    }
  )
  
  # 4. Retrieve Results
  final_study <- optuna$load_study(study_name = study_name, storage = storage_url)
  
  best_params_py <- tryCatch(final_study$best_params, error = function(e) list())
  best_value     <- tryCatch(final_study$best_value, error = function(e) -Inf)
  
  # CRITICAL FIX: Convert Python types to R types
  # This avoids issues where 'numpy.float64' causes errors in R math/serialization
  best_params_r <- lapply(best_params_py, function(x) {
    if (inherits(x, c("python.builtin.float", "numpy.float64", "numpy.float32"))) {
      return(as.numeric(x))
    } else if (inherits(x, c("python.builtin.int", "numpy.int64", "numpy.int32"))) {
      return(as.integer(x))
    } else if (inherits(x, c("python.builtin.str"))) {
      return(as.character(x))
    } else {
      return(x)
    }
  })
  
  if (verbose) message(sprintf("Optuna finished. Best Score: %.4f", best_value))
  
  structure(
    list(
      best_params = best_params_r,
      best_score  = best_value,
      n_trials    = n_trials,
      study       = final_study
    ),
    class = "fastsvm_optuna"
  )
}

# ------------------------------------------------------------------------------
#  Multi-Kernel Wrapper (Matches random_machines design)
# ------------------------------------------------------------------------------

#' Multi-Kernel Hyperparameter Tuning via Optuna
#'
#' Optimized version of \code{tune_random_machines} that uses Bayesian Optimization (Optuna).
#' It iterates over a list of kernel configurations and runs a parallelized 
#' Optuna search for each one to find the best hyperparameters.
#'
#' @details
#' This function is the "Bridge" between the raw tuning and the ensemble training.
#' It allows you to tune standard kernels (like "linear", "rbf") and custom
#' function-based kernels (like Wavelet) in a single call.
#' 
#' The results can be passed to \code{\link{as_kernels}} to generate the final
#' list of kernels for \code{\link{random_machines}}.
#'
#' @param data A \code{data.frame} containing training data.
#' @param time_col Name of the time column.
#' @param delta_col Name of the event column.
#' @param kernel_mix A named list of base kernel configurations. This serves as the 
#'   baseline. Keys must match those in \code{search_spaces}.
#' @param search_spaces A named list of search spaces. Each element should be a list 
#'   of parameters defined using \code{\link{opt_float}}, \code{\link{opt_int}}, etc.
#' @param n_trials Integer. Total number of trials per kernel.
#' @param cv Integer. Number of CV folds.
#' @param seed Integer. Random seed.
#' @param verbose Integer/Logical. Verbosity level.
#' @param cores Integer. Number of parallel cores to use for the search.
#' @param ... Additional global arguments passed to all kernels.
#'
#' @return An object of class \code{"random_machines_tune_optuna"} which also
#'   inherits from \code{"random_machines_tune"} to ensure compatibility with
#'   generic functions like \code{as_kernels}.
#' 
#' @examples
#' \dontrun{
#' if (reticulate::py_module_available("optuna") && requireNamespace("parallel")) {
#'   library(FastSurvivalSVM)
#'   
#'   # =========================================================================
#'   # 1. Data Generation
#'   # =========================================================================
#'   set.seed(42)
#'   df_train <- data_generation(n = 250, prop_cen = 0.25)
#'   df_test  <- data_generation(n = 100, prop_cen = 0.25)
#'   
#'   # =========================================================================
#'   # 2. Define Custom Kernel Functions (Math Only)
#'   # =========================================================================
#'   
#'   # Custom Polynomial Kernel: (x'z + coef0)^degree
#'   my_poly <- function(x, z, degree, coef0) {
#'     (sum(x * z) + coef0)^degree
#'   }
#'   
#'   # Custom Wavelet Kernel
#'   my_wavelet <- function(x, z, A) {
#'     u <- (as.numeric(x) - as.numeric(z)) / A
#'     prod(cos(1.75 * u) * exp(-0.5 * u^2))
#'   }
#'   
#'   # =========================================================================
#'   # 3. Tuning Workflow (Full Pipeline)
#'   # =========================================================================
#'   
#'   # A. Base Configurations (Fixed settings)
#'   # rank_ratio = 0.0 implies Regression mode (Learning Survival Time)
#'   kernel_mix <- list(
#'     linear_std = list(kernel = "linear", rank_ratio = 0.0),
#'     rbf_std    = list(kernel = "rbf",    rank_ratio = 0.0),
#'     poly_my    = list(kernel = my_poly,  rank_ratio = 0.0),
#'     wave_my    = list(kernel = my_wavelet, rank_ratio = 0.0)
#'   )
#'   
#'   # B. Search Spaces (Ranges for Optuna)
#'   search_spaces <- list(
#'     # Tune 'alpha' for Linear
#'     linear_std = list(
#'       alpha = opt_float(0.01, 10, log = TRUE)
#'     ),
#'     # Tune 'alpha' and 'gamma' for RBF
#'     rbf_std = list(
#'       alpha = opt_float(0.01, 10, log = TRUE),
#'       gamma = opt_float(0.001, 1, log = TRUE)
#'     ),
#'     # Tune 'alpha', 'degree', 'coef0' for Custom Poly
#'     poly_my = list(
#'       alpha  = opt_float(0.01, 10, log = TRUE),
#'       degree = opt_int(2, 4),
#'       coef0  = opt_float(0, 2)
#'     ),
#'     # Tune 'alpha' and 'A' for Custom Wavelet
#'     wave_my = list(
#'       alpha = opt_float(0.01, 10, log = TRUE),
#'       A     = opt_float(0.5, 2.5)
#'     )
#'   )
#'   
#'   # C. Run Optuna Optimization using ALL available cores
#'   #    - Spawns workers on all cores.
#'   #    - Each worker connects to a shared SQLite DB to avoid conflicts.
#'   #    - Kernels are tuned sequentially, but trials run in parallel.
#'   cat("Starting Optuna Tuning...\n")
#'   tune_res <- tune_random_machines_optuna(
#'     data          = df_train,
#'     time_col      = "tempo",
#'     delta_col     = "cens",
#'     kernel_mix    = kernel_mix,
#'     search_spaces = search_spaces,
#'     n_trials      = 20,    # 20 trials per kernel
#'     cv            = 3,     # 3-fold CV inside each trial
#'     cores         = parallel::detectCores(), # Use all cores
#'     seed          = 123
#'   )
#'   
#'   print(tune_res)
#'   
#'   # =========================================================================
#'   # 4. Train Random Machines (Ensemble)
#'   # =========================================================================
#'   
#'   # D. Extract best parameters found by Optuna
#'   final_kernels <- as_kernels(tune_res, kernel_mix)
#'   
#'   cat("Training Random Machines with Optimized Kernels...\n")
#'   model_rm <- random_machines(
#'     data         = df_train,
#'     newdata      = df_test,
#'     time_col     = "tempo",
#'     delta_col    = "cens",
#'     kernels      = final_kernels, # Use the tuned parameters
#'     B            = 50,            # Number of bootstrap samples
#'     prop_holdout = 0.2,           # Internal holdout for weights
#'     cores        = parallel::detectCores() # Parallel training
#'   )
#'   
#'   # =========================================================================
#'   # 5. Evaluate
#'   # =========================================================================
#'   print(model_rm)
#'   
#'   cidx <- score(model_rm, df_test)
#'   cat(sprintf("Final Test C-Index: %.4f\n", cidx))
#' }
#' }
#' @export
tune_random_machines_optuna <- function(
    data,
    time_col = "t",
    delta_col = "delta",
    kernel_mix,
    search_spaces,
    n_trials = 50L,
    cv = 5L,
    seed = NULL,
    verbose = 1L,
    cores = parallel::detectCores(),
    ...
) {
  
  common_names <- intersect(names(kernel_mix), names(search_spaces))
  if (length(common_names) == 0) stop("No matching names between 'kernel_mix' and 'search_spaces'.")
  
  results <- list()
  has_cli <- requireNamespace("cli", quietly = TRUE)
  
  if (verbose > 0 && has_cli) {
    cli::cli_h1("Random Machines Optuna Tuning")
    cli::cli_alert_info("Optimizing {length(common_names)} kernels ({n_trials} trials each).")
    cli::cli_alert_info("Strategy: Parallel Search ({cores} cores) | Serial CV.")
  }
  
  for (kname in common_names) {
    if (verbose > 0 && has_cli) cli::cli_h2(paste("Kernel:", kname))
    
    # Prepare arguments
    base_conf <- kernel_mix[[kname]]
    space     <- search_spaces[[kname]]
    
    # Remove search keys from base_conf if they accidentally exist there
    for (nm in names(space)) base_conf[[nm]] <- NULL
    
    # Combine call arguments
    call_args <- c(
      list(
        data = data, time_col = time_col, delta_col = delta_col,
        search_space = space,
        n_trials = n_trials, cv = cv, cores = cores, seed = seed,
        verbose = (verbose > 1) 
      ),
      base_conf,
      list(...)
    )
    
    # Execute
    res <- tryCatch({
      do.call(tune_fastsvm_optuna, call_args)
    }, error = function(e) {
      if (has_cli) cli::cli_alert_danger("Failed: {e$message}")
      NULL
    })
    
    if (verbose > 0 && has_cli && !is.null(res)) {
      if (res$best_score == -1) {
        cli::cli_alert_danger("All trials failed (Score = -1). Check constraints.")
      } else {
        cli::cli_alert_success("Best C-index: {.val {round(res$best_score, 4)}}")
      }
    }
    
    results[[kname]] <- res
  }
  
  # IMPORTANT: Dual inheritance for compatibility with as_kernels generics
  structure(results, class = c("random_machines_tune_optuna", "random_machines_tune"))
}

#' Print method for Optuna results
#'
#' Formats and prints the results of the multi-kernel optimization.
#'
#' @param x An object of class \code{"random_machines_tune_optuna"}.
#' @param ... Additional arguments.
#' @export
print.random_machines_tune_optuna <- function(x, ...) {
  if (requireNamespace("cli", quietly = TRUE)) {
    cli::cli_h1("Optuna Tuning Results")
    df <- do.call(rbind, lapply(names(x), function(nm) {
      obj <- x[[nm]]
      if (is.null(obj)) return(data.frame(Kernel=nm, C_Index=NA, Trials=0))
      data.frame(
        Kernel = nm,
        C_Index = obj$best_score,
        Trials = obj$n_trials
      )
    }))
    df <- df[order(df$C_Index, decreasing = TRUE), ]
    print(df, row.names = FALSE)
  } else {
    print(unclass(x))
  }
  invisible(x)
}

# ==============================================================================
#  Extract Kernels (Generic & Methods)
# ==============================================================================

#' Generic function for extracting best kernels
#'
#' Defines the generic function \code{as_kernels} to extract optimized kernel
#' configurations from tuning results.
#'
#' @param tune_results Tuning result object.
#' @param kernel_mix Original kernel configuration.
#' @export
as_kernels <- function(tune_results, kernel_mix) {
  UseMethod("as_kernels")
}

#' Extract Best Kernels from Optuna Results
#'
#' Helper function to merge the original kernel configuration with the 
#' best hyperparameters found by Optuna.
#'
#' @param tune_results The object returned by \code{tune_random_machines_optuna}.
#' @param kernel_mix The original list of base kernel configurations.
#'
#' @return A list of optimized kernel configurations ready for \code{random_machines}.
#' @export
as_kernels.random_machines_tune_optuna <- function(tune_results, kernel_mix) {
  tune_names <- names(tune_results)
  final_list <- list()
  
  # Standard SVM params that SHOULD NOT be stripped even if present in function
  # (Gamma/degree/coef0 are handled by sklearn if passed, but A is not)
  valid_svm_args <- c("alpha", "rank_ratio", "fit_intercept", "max_iter", 
                      "verbose", "tol", "optimizer", "random_state", 
                      "gamma", "degree", "coef0", "kernel")
  
  for (kname in tune_names) {
    res <- tune_results[[kname]]
    base_conf <- kernel_mix[[kname]]
    
    if (is.null(res) || is.null(res$best_score) || res$best_score == -1) {
      warning(sprintf("Kernel '%s' failed tuning. Using base config.", kname))
      final_list[[kname]] <- base_conf
    } else {
      # 1. Merge params (res$best_params are already cleaned R types)
      merged_conf <- utils::modifyList(base_conf, res$best_params)
      
      # 2. Robust Function Modification for Custom Kernels
      if (is.function(merged_conf$kernel)) {
        k_func_orig <- merged_conf$kernel
        k_args_names <- setdiff(names(formals(k_func_orig)), c("x", "z"))
        
        # Extract values for these args from the merged config
        k_args_vals <- merged_conf[intersect(names(merged_conf), k_args_names)]
        
        if (length(k_args_vals) > 0) {
          # Make a copy of the function
          new_func <- k_func_orig
          
          # Inject tuned values directly as defaults into the function formals
          # This makes the function self-contained for serialization
          for (arg_name in names(k_args_vals)) {
            formals(new_func)[[arg_name]] <- k_args_vals[[arg_name]]
          }
          
          merged_conf$kernel <- new_func
          
          # CRITICAL FIX: Remove the injected custom parameters (like 'A') from the 
          # configuration list. If we don't, they are passed to the Python constructor
          # as **kwargs, causing "unexpected keyword argument" error.
          # We only keep args that are valid for FastKernelSurvivalSVM.
          args_to_remove <- setdiff(names(k_args_vals), valid_svm_args)
          if (length(args_to_remove) > 0) {
            merged_conf[args_to_remove] <- NULL
          }
        }
      }
      final_list[[kname]] <- merged_conf
    }
  }
  return(final_list)
}

#' Extract Best Kernels from Grid Search Results
#'
#' Helper function to merge the original kernel configuration with the 
#' best hyperparameters found by standard Grid Search (\code{tune_random_machines}).
#'
#' @param tune_results The object returned by \code{tune_random_machines}.
#' @param kernel_mix Original kernel configuration.
#' @export
as_kernels.random_machines_tune <- function(tune_results, kernel_mix) {
  tune_names <- names(tune_results)
  final_list <- list()
  
  for (kname in tune_names) {
    res <- tune_results[[kname]]
    base_conf <- kernel_mix[[kname]]
    
    if (is.null(res)) {
      final_list[[kname]] <- base_conf
    } else {
      # Merge base config with best tuned params
      final_list[[kname]] <- utils::modifyList(base_conf, res$best_params)
    }
  }
  return(final_list)
}