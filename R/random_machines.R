# R/random_machines.R

#' Generic function for computing concordance index (score)
#'
#' @param object A fitted model object.
#' @param data A data frame containing the validation data.
#' @param ... Additional arguments.
#' @export
score <- function(object, data, ...) {
  UseMethod("score")
}

# -------------------------------------------------------------------
# Helpers internos
# -------------------------------------------------------------------

#' Compute normalized weights from C-indices
#' @keywords internal
.compute_lambdas_from_cindex <- function(cindex_vec, beta = 1) {
  # Numerical protection
  # If C-index <= 0.5, treat as random/poor
  cindex_vec[is.na(cindex_vec) | cindex_vec <= 0.5] <- 0.500001
  cindex_vec[cindex_vec >= 1] <- 0.999999
  
  # Log-odds transform (conforme script serial random-survival-machines.R)
  num <- log(cindex_vec / (1 - cindex_vec))
  
  if (beta != 1) num <- num * beta
  
  lambdas <- num / sum(num)
  return(lambdas)
}

# -------------------------------------------------------------------
# Main Bagging Function
# -------------------------------------------------------------------

#' Parallel Bagging for FastKernelSurvivalSVM (Random Machines)
#'
#' Fits an ensemble of models using bootstrap aggregation (bagging) and
#' computes predictions for \code{newdata}. This function is also referred to
#' as "Random Machines" in the context of kernel survival machines.
#'
#' \strong{Internal Holdout for Kernel Weights:}
#' Instead of using training performance (resubstitution) to define the
#' selection probabilities of each kernel, this function splits the training
#' data into an internal training/validation set according to \code{prop_holdout}.
#' This mimics the behavior of the serial implementation where weights are
#' fixed based on a pre-bagging holdout.
#'
#' \strong{Architecture:}
#' This function adopts a "Train-and-Predict" strategy:
#' \itemize{
#'   \item It accepts \code{newdata} and computes predictions inside the
#'         parallel workers (for each bootstrap model).
#'   \item It stores a serialized version (Python pickle) of each
#'         fitted model to allow future predictions via
#'         \code{\link{predict.random_machines}}.
#' }
#'
#' \strong{Random Subspace (mtry):}
#' The \code{mtry} parameter allows for random selection of a subset of
#' covariates in each base learner, similar to Random Forests.
#'
#' @param data A \code{data.frame} containing training data.
#' @param newdata A \code{data.frame} containing test data for prediction.
#' @param time_col Name of the column with survival times.
#' @param delta_col Name of the column with the event indicator (1 = event, 0 = censored).
#' @param kernels A named list of kernel specifications. Each element must be a list
#'   of arguments to the estimator (e.g., \code{kernel}, \code{alpha},
#'   \code{rank_ratio}, \code{fit_intercept}, \code{max_iter}, etc.).
#' @param B Integer. Number of bootstrap samples.
#' @param mtry Integer or Numeric. Number of variables to randomly sample at each split.
#'   \itemize{
#'     \item \code{NULL} (default): Use all variables.
#'     \item Integer >= 1: Select exactly \code{mtry} variables.
#'     \item Numeric < 1: Select \code{mtry} fraction of variables (e.g., 0.5 = 50\%).
#'   }
#' @param crop Numeric or NULL. Threshold for kernel selection probabilities.
#'   If provided (e.g., \code{0.10}), any kernel with a calculated weight
#'   less than or equal to this value in the holdout phase will be
#'   discarded (weight set to 0), and the remaining weights will be
#'   rescaled to sum to 1.
#' @param beta_kernel Numeric. Temperature for kernel selection probabilities
#'   (based on internal holdout C-index).
#' @param beta_bag Numeric. Temperature for ensemble weighting
#'   (based on OOB C-index of each bootstrap model).
#' @param cores Integer. Number of parallel workers (via \code{mirai}).
#' @param seed Optional integer passed to \code{mirai::daemons} and used for
#'   internal sampling.
#' @param prop_holdout Numeric in (0, 1). Proportion of the original training data
#'   used as internal validation when computing the kernel selection weights.
#'   For example, \code{prop_holdout = 0.20} means that 20\% of the rows are
#'   used as validation and the remaining 80\% as training in the weighting phase.
#'   If the dataset is too small for this split (either side < 10 rows),
#'   the function falls back to resubstitution.
#' @param .progress Logical. Show progress bar for the bootstrap loop?
#'
#' @return An object of class \code{"random_machines"} containing:
#'   \itemize{
#'     \item \code{preds}: Numeric vector of aggregated predictions for \code{newdata}.
#'     \item \code{weights}: Vector of weights assigned to each bootstrap model.
#'     \item \code{chosen_kernels}: Vector of kernel names selected in each bootstrap.
#'     \item \code{c_indices}: Vector of OOB C-indices for each bootstrap.
#'     \item \code{rank_ratio}: Rank ratio stored from the first successful model.
#'     \item \code{time_col}, \code{delta_col}: Names of the survival columns.
#'     \item \code{ensemble}: List with serialized models (Python pickle), features and params.
#'     \item \code{mtry}: The mtry value used.
#'     \item \code{kernel_lambdas}: The kernel selection probabilities (fixed from holdout).
#'     \item \code{kernel_names}: The names of the kernels used in bagging.
#'     \item \code{prop_holdout}: The proportion used for the internal holdout split.
#'     \item \code{crop}: The crop threshold used.
#'   }
#'
#' @examples
#' \dontrun{
#' if (reticulate::py_module_available("sksurv") && requireNamespace("mirai")) {
#'   library(FastSurvivalSVM)
#'
#'   ## 1. Data generation
#'   set.seed(3)
#'   df <- data_generation(n = 300, prop_cen = 0.3)
#'
#'   ## Train/Test split
#'   idx <- sample(seq_len(nrow(df)), 200)
#'   train_df <- df[idx, ]
#'   test_df  <- df[-idx, ]
#'
#'   ## 2. Custom Kernel Factories ----------------------------------
#'   make_wavelet <- function(A = 1) {
#'     force(A)
#'     function(x, z) {
#'       u <- (as.numeric(x) - as.numeric(z)) / A
#'       prod(cos(1.75 * u) * exp(-0.5 * u^2))
#'     }
#'   }
#'
#'   make_poly <- function(degree = 3, coef0 = 1) {
#'     force(degree); force(coef0)
#'     function(x, z) (sum(as.numeric(x) * as.numeric(z)) + coef0)^degree
#'   }
#'
#'   ## 3. Kernel Specifications (rank_ratio = 0 for Regression / Time)
#'   kernel_mix <- list(
#'     linear   = list(
#'       kernel        = "linear",
#'       alpha         = 1,
#'       rank_ratio    = 0,
#'       fit_intercept = TRUE
#'     ),
#'     rbf      = list(
#'       kernel        = "rbf",
#'       alpha         = 0.5,
#'       gamma         = 0.1,
#'       rank_ratio    = 0,
#'       fit_intercept = TRUE
#'     ),
#'     poly_std = list(
#'       kernel        = "poly",
#'       degree        = 2L,
#'       alpha         = 1,
#'       rank_ratio    = 0,
#'       fit_intercept = TRUE
#'     ),
#'     wavelet  = list(
#'       kernel        = make_wavelet(A = 1),
#'       alpha         = 1,
#'       rank_ratio    = 0,
#'       fit_intercept = TRUE
#'     ),
#'     poly_fun = list(
#'       kernel        = make_poly(degree = 2L),
#'       alpha         = 1,
#'       rank_ratio    = 0,
#'       fit_intercept = TRUE
#'     )
#'   )
#'
#'   ## 4. Run Random Machines with internal holdout for kernel weights
#'   rm_results <- random_machines(
#'     data         = train_df,
#'     newdata      = test_df,
#'     time_col     = "tempo",
#'     delta_col    = "cens",
#'     kernels      = kernel_mix,
#'     B            = 50,
#'     mtry         = NULL,
#'     beta_kernel  = 1,
#'     beta_bag     = 1,
#'     cores        = parallel::detectCores(),
#'     seed         = 42,
#'     prop_holdout = 0.20,
#'     crop         = 0.15,  # Eliminate kernels with weight <= 0.15
#'     .progress    = TRUE
#'   )
#'
#'   print(rm_results)
#'
#'   ## 5. Score on independent test set using S3 method
#'   cidx_test <- score(rm_results, test_df)
#'   cat(sprintf("Test C-index (Random Machines): %.4f\n", cidx_test))
#' }
#' }
#'
#' @importFrom mirai daemons
#' @importFrom purrr map in_parallel
#' @importFrom reticulate py_run_string py_call import py_available py_config
#' @import cli
#' @export
random_machines <- function(
  data,
  newdata,
  time_col    = "t",
  delta_col   = "delta",
  kernels,
  B           = 100L,
  mtry        = NULL,
  crop        = NULL,
  beta_kernel = 1,
  beta_bag    = 1,
  cores       = 1L,
  seed        = NULL,
  prop_holdout = 0.20,
  .progress   = TRUE
) {
  # --- Basic validations ---
  stopifnot(is.data.frame(data))
  stopifnot(is.data.frame(newdata))
  
  if (!time_col  %in% names(data))  stop("`time_col` not found in `data`.")
  if (!delta_col %in% names(data))  stop("`delta_col` not found in `data`.")
  if (!is.list(kernels) || is.null(names(kernels)))
    stop("`kernels` must be a named list.")
  
  # Validate columns in newdata
  feature_cols <- setdiff(names(data), c(time_col, delta_col))
  missing_cols <- setdiff(feature_cols, names(newdata))
  if (length(missing_cols) > 0) {
    stop("`newdata` is missing columns found in `data`: ",
         paste(missing_cols, collapse = ", "))
  }
  
  B <- as.integer(B)
  if (B <= 0L) stop("`B` must be a positive integer.")
  
  if (!requireNamespace("mirai", quietly = TRUE)) 
    stop("Package 'mirai' required.")
  if (!requireNamespace("purrr", quietly = TRUE)) 
    stop("Package 'purrr' required.")
  if (!requireNamespace("reticulate", quietly = TRUE))
    stop("Package 'reticulate' required.")
  if (!requireNamespace("cli", quietly = TRUE))
    stop("Package 'cli' required.")
  
  # Verificação opcional do pacote emo
  has_emo <- requireNamespace("emo", quietly = TRUE)
  ji <- function(x, fallback = "") if (has_emo) emo::ji(x) else fallback
  
  if (!is.null(seed)) set.seed(seed)
  
  n            <- nrow(data)
  kernel_names <- names(kernels)
  
  # ------------------------------------------------------------------
  # 1. Internal Holdout for Kernel Weights
  # ------------------------------------------------------------------
  
  # Cabeçalho estilizado
  cli::cli_h1(paste(ji("rocket", ">>"), "Random Machines (Kernel Survival SVM)"))
  cli::cli_alert_info("Starting Random Machines (B={B}, mtry={if (is.null(mtry)) 'All' else mtry}) on {cores} cores.")
  
  n_val <- floor(prop_holdout * n)
  use_holdout <- (n_val >= 10L && (n - n_val) >= 10L)
  
  if (use_holdout) {
    idx_val   <- sample(seq_len(n), size = n_val)
    d_train_w <- data[-idx_val, , drop = FALSE]
    d_val_w   <- data[idx_val,  , drop = FALSE]
    
    cli::cli_alert_info(
      "Kernel weights via Holdout: {.strong {nrow(d_train_w)}} training / {.strong {n_val}} validation."
    )
  } else {
    cli::cli_alert_warning(
      "Not enough data for holdout. Using {.strong resubstitution} for kernel weights."
    )
    d_train_w <- data
    d_val_w   <- data
    n_val     <- n
  }
  
  # Silence Python warnings globally for this phase
  if (reticulate::py_available(initialize = TRUE)) {
    try(
      reticulate::py_run_string(
        "import warnings; warnings.simplefilter('ignore')"
      ),
      silent = TRUE
    )
  }
  
  base_cindex <- numeric(length(kernel_names))
  names(base_cindex) <- kernel_names
  
  # Calculando pesos
  cli::cli_alert_info("Computing kernel weights...")
  
  for (i in seq_along(kernel_names)) {
    kname <- kernel_names[i]
    spec  <- kernels[[kname]]
    
    # Defaults consistent with bagging
    if (is.null(spec$rank_ratio))    spec$rank_ratio    <- 0
    if (is.null(spec$fit_intercept)) spec$fit_intercept <- TRUE
    
    args_fit <- c(
      list(
        data      = d_train_w,
        time_col  = time_col,
        delta_col = delta_col
      ),
      spec
    )
    
    # Silent failure -> neutral C-index = 0.5
    base_cindex[i] <- tryCatch(
      {
        mod <- do.call(fastsvm, args_fit)
        score(mod, d_val_w)
      },
      error = function(e) 0.5
    )
  }
  
  kernel_lambdas <- .compute_lambdas_from_cindex(base_cindex, beta = beta_kernel)
  
  # ------------------------------------------------------------------
  # 1.b Apply CROP logic
  # ------------------------------------------------------------------
  if (!is.null(crop)) {
    # Máscara de quem sobrevive
    keep_mask <- kernel_lambdas > crop
    n_kept    <- sum(keep_mask)
    n_total   <- length(kernel_lambdas)
    
    if (n_kept == 0) {
      # Fallback de segurança: se todos forem eliminados, mantém o melhor
      best_idx <- which.max(kernel_lambdas)
      keep_mask[best_idx] <- TRUE
      cli::cli_alert_warning(
        "All kernels had weights <= crop ({crop}). Keeping the single best one: {.strong {names(kernel_lambdas)[best_idx]}}."
      )
    } else if (n_kept < n_total) {
      cli::cli_alert_info(
        "Crop applied: {n_total - n_kept} kernel(s) dropped (weight <= {crop}). Re-scaling weights."
      )
    }
    
    # Zerar os eliminados
    kernel_lambdas[!keep_mask] <- 0
    
    # Re-normalizar os sobreviventes para somar 1
    if (sum(kernel_lambdas) > 0) {
      kernel_lambdas <- kernel_lambdas / sum(kernel_lambdas)
    }
  }
  
  # Capture Python path from main session
  py_path_main <- tryCatch(reticulate::py_config()$python, error = function(e) NULL)
  
  # ------------------------------------------------------------------
  # 2. Parallel setup (Mirai)
  # ------------------------------------------------------------------
  if (cores > 1L) {
    mirai::daemons(n = cores, seed = seed, dispatcher = TRUE)
    on.exit(mirai::daemons(0L), add = TRUE)
  } else if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # ------------------------------------------------------------------
  # 3. Parallel Bootstrap (Train & Predict in workers)
  # ------------------------------------------------------------------
  cli::cli_alert_info("Executing parallel bootstrap...")
  
  boot_results <- purrr::map(
    .x = seq_len(B),
    .f = purrr::in_parallel(
      function(b) {
        # --- A. Worker setup ---
        library(reticulate)
        
        # Ensure correct Python and silence warnings
        if (!is.null(python_path_ref)) {
          Sys.setenv(RETICULATE_PYTHON = python_path_ref)
        }
        if (!reticulate::py_available(initialize = TRUE)) {
          reticulate::py_config()
        }
        try(
          reticulate::py_run_string(
            "import warnings; warnings.simplefilter('ignore')"
          ),
          silent = TRUE
        )
        
        # Local imports
        sksvm_loc  <- reticulate::import("sksurv.svm")
        sksurv_loc <- reticulate::import("sksurv")
        pickle_loc <- reticulate::import("pickle")
        
        local_mk_surv <- function(t, e) {
          sksurv_loc$util$Surv$from_arrays(
            event = as.logical(e),
            time  = as.numeric(t)
          )
        }
        
        # --- B. Bagging logic ---
        # A função sample lida automaticamente com prob=0 (nunca sorteia)
        # desde que a soma das probs seja > 0 (garantido pelo re-scaling)
        kname <- sample(kernel_names_ref, size = 1, prob = kernel_lambdas_ref)
        spec  <- kernels_ref[[kname]]
        
        # Row bootstrap
        boo_index <- sample.int(n_ref, n_ref, replace = TRUE)
        oob_index <- setdiff(seq_len(n_ref), unique(boo_index))
        
        df_boot <- data_ref[boo_index, , drop = FALSE]
        mask_ok <- is.finite(df_boot[[time_col_ref]]) & !is.na(df_boot[[time_col_ref]])
        df_boot <- df_boot[mask_ok, , drop = FALSE]
        
        n_test  <- nrow(newdata_ref)
        na_pred <- rep(NA_real_, n_test)
        
        if (nrow(df_boot) < 10L) {
          return(NULL)
        }
        
        # Feature selection (mtry)
        all_features <- setdiff(names(df_boot), c(time_col_ref, delta_col_ref))
        
        if (!is.null(mtry_ref)) {
          n_feat <- length(all_features)
          n_sel  <- if (mtry_ref < 1) {
            ceiling(n_feat * mtry_ref)
          } else {
            min(n_feat, as.integer(mtry_ref))
          }
          n_sel  <- max(1L, n_sel)
          x_cols <- sample(all_features, size = n_sel)
        } else {
          x_cols <- all_features
        }
        
        X_mat  <- as.matrix(df_boot[, x_cols, drop = FALSE])
        y_surv <- local_mk_surv(df_boot[[time_col_ref]], df_boot[[delta_col_ref]] == 1)
        
        # Newdata matrix
        if (!all(x_cols %in% names(newdata_ref))) {
          return(NULL)
        }
        X_test <- as.matrix(newdata_ref[, x_cols, drop = FALSE])
        
        # Configure params
        params <- spec
        k_arg  <- params$kernel
        params$kernel <- NULL
        
        if (is.function(k_arg)) {
          k_py <- function(x, z) k_arg(as.numeric(x), as.numeric(z))
        } else {
          k_py <- k_arg
        }
        
        if (is.null(params$fit_intercept)) params$fit_intercept <- TRUE
        if (is.null(params$rank_ratio))    params$rank_ratio    <- 0.0
        if (is.null(params$max_iter))      params$max_iter      <- 1000L
        
        final_args <- c(list(kernel = k_py), params)
        
        # Fit model
        model <- tryCatch({
          mod <- do.call(sksvm_loc$FastKernelSurvivalSVM, final_args)
          mod$fit(X_mat, y_surv)
          mod
        }, error = function(e) NULL)
        
        if (is.null(model)) {
          return(NULL)
        }
        
        # OOB C-index
        c_index_b <- 0.5
        if (length(oob_index) > 0L) {
          dados_oob <- data_ref[oob_index, , drop = FALSE]
          mask      <- is.finite(dados_oob[[time_col_ref]])
          if (sum(mask) > 0L) {
            X_oob <- as.matrix(dados_oob[mask, x_cols, drop = FALSE])
            y_oob <- local_mk_surv(
              dados_oob[mask, ][[time_col_ref]],
              dados_oob[mask, ][[delta_col_ref]] == 1
            )
            c_index_b <- tryCatch(
              as.numeric(model$score(X_oob, y_oob)),
              error = function(e) 0.5
            )
          }
        }
        
        # Predictions on newdata
        pred_vec <- tryCatch(
          as.numeric(model$predict(X_test)),
          error = function(e) na_pred
        )
        
        # Serialize (pickle) for future use
        model_bytes <- tryCatch({
          py_bytes <- reticulate::py_call(pickle_loc$dumps, model)
          as.raw(reticulate::py_to_r(py_bytes))
        }, error = function(e) NULL)
        
        list(
          c_index_b   = c_index_b,
          pred_vec    = pred_vec,
          kname       = kname,
          rank_ratio  = params$rank_ratio,
          model_bytes = model_bytes,
          x_cols      = x_cols,
          params      = spec
        )
      },
      # Exports
      data_ref           = data,
      newdata_ref        = newdata,
      time_col_ref       = time_col,
      delta_col_ref      = delta_col,
      n_ref              = n,
      kernel_names_ref   = kernel_names,
      kernel_lambdas_ref = kernel_lambdas,
      kernels_ref        = kernels,
      python_path_ref    = py_path_main,
      mtry_ref           = mtry
    ),
    .progress = .progress
  )
  
  # ------------------------------------------------------------------
  # 4. Aggregation
  # ------------------------------------------------------------------
  boot_results <- Filter(Negate(is.null), boot_results)
  if (length(boot_results) == 0L) {
    cli::cli_alert_danger("Random Machines failed completely: no valid bootstrap results.")
    stop("No valid models produced.")
  }
  
  successes <- boot_results
  
  c_indices      <- vapply(successes, `[[`, numeric(1),   "c_index_b")
  chosen_kernels <- vapply(successes, `[[`, character(1), "kname")
  
  boot_lambdas <- .compute_lambdas_from_cindex(c_indices, beta = beta_bag)
  
  preds_list  <- lapply(successes, `[[`, "pred_vec")
  n_test      <- nrow(newdata)
  final_pred  <- numeric(n_test)
  total_weight <- 0
  
  for (i in seq_along(preds_list)) {
    p <- preds_list[[i]]
    w <- boot_lambdas[i]
    if (!any(is.na(p))) {
      final_pred  <- final_pred + (p * w)
      total_weight <- total_weight + w
    }
  }
  
  if (total_weight > 0) {
    final_pred <- final_pred / total_weight
  } else {
    final_pred <- rep(NA_real_, n_test)
  }
  
  ensemble_light <- lapply(successes, function(x) {
    list(
      model_bytes = x$model_bytes,
      x_cols      = x$x_cols,
      params      = x$params
    )
  })
  
  rr <- successes[[1]]$rank_ratio
  mean_oob <- mean(c_indices, na.rm = TRUE)
  
  cli::cli_alert_success(
    "Done. Valid Models: {.strong {length(successes)}}/{B}. Mean OOB: {.val {round(mean_oob, 4)}}"
  )
  
  structure(
    list(
      preds          = final_pred,
      weights        = boot_lambdas,
      chosen_kernels = chosen_kernels,
      c_indices      = c_indices,
      rank_ratio     = rr,
      time_col       = time_col,
      delta_col      = delta_col,
      ensemble       = ensemble_light,
      mtry           = mtry,
      kernel_lambdas = kernel_lambdas,
      kernel_names   = kernel_names,
      prop_holdout   = prop_holdout,
      crop           = crop
    ),
    class = "random_machines"
  )
}

# -------------------------------------------------------------------
# Score & Print
# -------------------------------------------------------------------

#' Score method for Random Machines
#'
#' Computes the concordance index for the aggregated predictions.
#' 
#' @param object An object of class \code{"random_machines"}.
#' @param data Validation data (must contain time/event cols).
#' @param ... Not used.
#' @export
score.random_machines <- function(object, data, ...) {
  stopifnot(inherits(object, "random_machines"))
  
  time  <- data[[object$time_col]]
  event <- data[[object$delta_col]]
  preds <- object$preds
  
  if (length(preds) != nrow(data)) {
    stop("Mismatch: Predictions in object do not match rows in 'data'.")
  }
  
  if (!requireNamespace("survival", quietly = TRUE))
    stop("Need package 'survival'.")
  
  survival::concordance(
    survival::Surv(time, event) ~ preds
  )$concordance
}

#' Print method for random_machines
#' @export
print.random_machines <- function(x, ...) {
  if (!requireNamespace("cli", quietly = TRUE)) {
    print.default(x)
    return(invisible(x))
  }
  
  has_emo <- requireNamespace("emo", quietly = TRUE)
  ji <- function(x, fallback = "") if (has_emo) emo::ji(x) else fallback
  
  # Configurar Tema
  d <- cli::cli_div(theme = list(
    span.emph = list(color = "cyan"), 
    span.strong = list(color = "blue", font_weight = "bold"),
    span.val = list(color = "green"),
    rule = list(color = "grey")
  ))
  
  cat("\n")
  cli::cli_h1(paste(ji("package", "#"), "Random Machines (FastKernelSurvivalSVM)"))
  
  cli::cli_ul()
  cli::cli_li("Models Trained: {.strong {length(x$weights)}}")
  cli::cli_li("Features (mtry): {.emph {if (is.null(x$mtry)) 'All' else x$mtry}}")
  cli::cli_li("Mean OOB C-index: {.val {round(mean(x$c_indices, na.rm = TRUE), 4)}}")
  if (!is.null(x$crop)) {
    cli::cli_li("Crop Threshold: {.strong {x$crop}}")
  }
  cli::cli_end()
  
  cat("\n")
  
  # --- Tabela 1: Kernel Usage (Bootstrap Selection) ---
  cli::cli_h2(paste(ji("bar_chart", "||"), "Kernel Usage (Bootstrap Selection)"))
  
  tbl   <- table(x$chosen_kernels)
  props <- prop.table(tbl)
  
  df_usage <- data.frame(
    Kernel = names(tbl),
    Count  = as.integer(tbl),
    Prob   = as.numeric(props), 
    stringsAsFactors = FALSE
  )
  df_usage <- df_usage[order(-df_usage$Count), ]
  
  # ALINHAMENTO DINÂMICO
  # Calcula a largura máxima de cada coluna considerando Cabeçalho e Dados
  w_kern_u  <- max(nchar(df_usage$Kernel), nchar("Kernel"))
  w_count_u <- max(nchar(as.character(df_usage$Count)), nchar("Count"))
  w_prob_u  <- max(nchar("Probability"), nchar("0.0000")) # Min 11 para "Probability"
  
  cli::cli_rule()
  cat(sprintf(
    "%-*s | %*s | %*s\n", 
    w_kern_u, cli::style_bold("Kernel  "), 
    w_count_u, cli::style_bold("Count"), 
    w_prob_u, cli::style_bold("Probability")
  ))
  cli::cli_rule()
  
  for (i in seq_len(nrow(df_usage))) {
    count_fmt <- sprintf("%*d", w_count_u, df_usage$Count[i])
    prob_fmt  <- sprintf("%*.4f", w_prob_u, df_usage$Prob[i])
    
    if (df_usage$Prob[i] > 0.2) {
      prob_fmt <- cli::col_green(prob_fmt)
    }
    
    cat(sprintf(
      "%-*s | %s | %s\n",
      w_kern_u, df_usage$Kernel[i], count_fmt, prob_fmt
    ))
  }
  cli::cli_rule()
  
  cat("\n")
  
  # --- Tabela 2: Kernel Weights (Holdout Probabilities) ---
  cli::cli_h2(paste(ji("balance_scale", "||"), "Kernel Weights (Holdout Probabilities)"))
  
  df_w <- data.frame(
    Kernel = names(x$kernel_lambdas),
    Prob   = as.numeric(x$kernel_lambdas),
    stringsAsFactors = FALSE
  )
  # Ordenar: maiores probs primeiro
  df_w <- df_w[order(-df_w$Prob), ]
  
  # Larguras dinâmicas para Tabela 2
  w_kern_w <- max(nchar(df_w$Kernel), nchar("Kernel"))
  w_prob_w <- max(nchar("Probability"), nchar("0.0000"))
  # Status width não precisa de calculo rigido pois é a ultima coluna
  
  cli::cli_rule()
  cat(sprintf(
    "%-*s | %*s | %s\n", 
    w_kern_w, cli::style_bold("Kernel  "), 
    w_prob_w, cli::style_bold("Probability"), 
    cli::style_bold("Status")
  ))
  cli::cli_rule()
  
  for (i in seq_len(nrow(df_w))) {
    prob_val <- df_w$Prob[i]
    prob_fmt <- sprintf("%*.4f", w_prob_w, prob_val)
    
    if (prob_val > 0) {
      # Texto "Selected" com check verde
      status_icon <- if(has_emo) ji("white_check_mark") else "OK"
      prob_fmt    <- cli::col_green(prob_fmt)
      status_full <- paste(status_icon, cli::style_bold(cli::col_green("Selected")))
    } else {
      # Texto "Eliminated" com x vermelho
      status_icon <- if(has_emo) ji("x") else "X"
      prob_fmt    <- cli::col_red(prob_fmt)
      status_full <- paste(status_icon, cli::style_bold(cli::col_red("Eliminated")))
    }
    
    cat(sprintf(
      "%-*s | %s | %s\n",
      w_kern_w, df_w$Kernel[i], prob_fmt, status_full
    ))
  }
  cli::cli_rule()
  
  cli::cli_end(d)
  cat("\n")
  
  invisible(x)
}

# -------------------------------------------------------------------
# Predict method (using stored ensemble)
# -------------------------------------------------------------------

#' Predict method for Random Machines
#'
#' Uses serialized models (Python pickle) to compute predictions for new data.
#'
#' @param object An object of class \code{"random_machines"}.
#' @param newdata A \code{data.frame} with covariates compatible with training data.
#' @param ... Not used, for compatibility.
#' @export
predict.random_machines <- function(object, newdata, ...) {
  stopifnot(inherits(object, "random_machines"))
  stopifnot(is.data.frame(newdata))
  
  tryCatch({
    reticulate::import("sksurv.svm")
    reticulate::import("numpy")
    pickle <- reticulate::import("pickle")
  }, error = function(e) stop("Need python/pickle."))
  
  M          <- length(object$ensemble)
  n_new      <- nrow(newdata)
  final_preds <- numeric(n_new)
  total_weight <- 0
  
  for (m in seq_len(M)) {
    item <- object$ensemble[[m]]
    if (is.null(item$model_bytes)) next
    
    weight <- object$weights[m]
    
    # Check model-specific columns (mtry)
    if (!all(item$x_cols %in% names(newdata))) next
    
    X_new <- as.matrix(newdata[, item$x_cols, drop = FALSE])
    
    py_mod <- tryCatch(pickle$loads(item$model_bytes), error = function(e) NULL)
    
    k_orig <- item$params$kernel
    if (is.function(k_orig) && !is.null(py_mod)) {
      k_py        <- function(x, z) k_orig(as.numeric(x), as.numeric(z))
      py_mod$kernel <- k_py
    }
    
    if (!is.null(py_mod)) {
      p <- tryCatch(
        as.numeric(py_mod$predict(X_new)),
        error = function(e) rep(NA_real_, n_new)
      )
      if (!all(is.na(p))) {
        final_preds  <- final_preds + (p * weight)
        total_weight <- total_weight + weight
      }
    }
  }
  
  if (total_weight == 0) return(rep(NA_real_, n_new))
  return(final_preds / total_weight)
}