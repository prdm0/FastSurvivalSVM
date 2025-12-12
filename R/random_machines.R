# R/random_machines.R

# Declare global variables to avoid R CMD check notes
utils::globalVariables(c(
  "kernel_names_ref", "kernels_ref", "kernel_lambdas_ref",
  "n_ref", "data_ref", "time_col_ref", "newdata_ref", "delta_col_ref",
  "mtry_ref", "python_path_ref"
))

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
  cindex_vec[is.na(cindex_vec) | cindex_vec <= 0.5] <- 0.500001
  cindex_vec[cindex_vec >= 1] <- 0.999999
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
#' Fits an ensemble of models using bootstrap aggregation (bagging).
#'
#' @param data A \code{data.frame} containing training data.
#' @param newdata A \code{data.frame} containing test data for prediction.
#' @param time_col Name of the column with survival times.
#' @param delta_col Name of the column with the event indicator (1 = event, 0 = censored).
#' @param kernels A named list of kernel specifications.
#' @param B Integer. Number of bootstrap samples.
#' @param mtry Integer or Numeric. Number of variables to randomly sample at each split.
#' @param crop Numeric or NULL. Threshold for kernel selection probabilities.
#' @param beta_kernel Numeric. Temperature for kernel selection probabilities.
#' @param beta_bag Numeric. Temperature for ensemble weighting.
#' @param cores Integer. Number of parallel workers (via \code{mirai}).
#' @param seed Optional integer passed to \code{mirai::daemons}.
#' @param prop_holdout Numeric in (0, 1). Proportion for internal holdout.
#' @param .progress Logical. Show progress bar?
#'
#' @return An object of class \code{"random_machines"}.
#'
#' @examples
#' \dontrun{
#' if (reticulate::py_module_available("sksurv") && requireNamespace("mirai")) {
#'   library(FastSurvivalSVM)
#'
#'   # 1. Data Generation and Split
#'   set.seed(42)
#'   df <- data_generation(n = 250, prop_cen = 0.25)
#'   train_idx <- sample(nrow(df), 200)
#'   train_df  <- df[train_idx, ]
#'   test_df   <- df[-train_idx, ]
#'
#'   # 2. Define Custom Kernel Functions (Math Only)
#'   
#'   # Wavelet Kernel
#'   my_wavelet <- function(x, z, A) {
#'     u <- (as.numeric(x) - as.numeric(z)) / A
#'     prod(cos(1.75 * u) * exp(-0.5 * u^2))
#'   }
#'
#'   # Polynomial Kernel
#'   my_poly <- function(x, z, degree, coef0) {
#'     (sum(as.numeric(x) * as.numeric(z)) + coef0)^degree
#'   }
#'
#'   # 3. Tuning Workflow
#'   #    Before training the ensemble, we optimize the hyperparameters for each
#'   #    kernel family using 'tune_random_machines'.
#'
#'   # A. Define Kernel Mix (Fixed Structure)
#'   #    We set rank_ratio = 0 because we want to solve a Regression problem.
#'   kernel_mix <- list(
#'     linear_std = list(kernel = "linear", rank_ratio = 0),
#'     rbf_std    = list(kernel = "rbf",    rank_ratio = 0),
#'     wavelet_ok = list(rank_ratio = 0),
#'     poly_ok    = list(rank_ratio = 0)
#'   )
#'
#'   # B. Define Parameter Grids (Search Space)
#'   #    We define 4 values for each hyperparameter to be tuned.
#'   param_grids <- list(
#'     # Linear: Tune regularization (alpha)
#'     linear_std = list(
#'       alpha = c(0.01, 0.1, 1.0, 10.0)
#'     ),
#'
#'     # RBF: Tune alpha and kernel width (gamma)
#'     rbf_std = list(
#'       alpha = c(0.01, 0.1, 1.0, 10.0),
#'       gamma = c(0.001, 0.01, 0.1, 1.0)
#'     ),
#'
#'     # Custom Wavelet: Tune alpha and the kernel parameter 'A'
#'     # 'grid_kernel' generates the variants for the custom function.
#'     wavelet_ok = list(
#'       kernel = grid_kernel(my_wavelet, A = c(0.5, 1.0, 1.5, 2.0)),
#'       alpha  = c(0.01, 0.1, 1.0, 10.0)
#'     ),
#'
#'     # Custom Poly: Tune alpha and the degree
#'     # (coef0 kept fixed at 1 for this example)
#'     poly_ok = list(
#'       kernel = grid_kernel(my_poly, degree = c(2, 3, 4, 5), coef0 = 1),
#'       alpha  = c(0.01, 0.1, 1.0, 10.0)
#'     )
#'   )
#'
#'   # C. Execute Hybrid Tuning
#'   #    This uses Python threads for Native kernels and R processes for Custom ones.
#'   cat("Starting hyperparameter tuning...\n")
#'   tune_res <- tune_random_machines(
#'     data        = train_df,
#'     time_col    = "tempo",
#'     delta_col   = "cens",
#'     kernel_mix  = kernel_mix,
#'     param_grids = param_grids,
#'     cv          = 3,
#'     cores       = parallel::detectCores(),
#'     verbose     = 1
#'   )
#'   
#'   # D. Bridge: Extract Best Hyperparameters
#'   #    This creates the final configuration list ready for the ensemble.
#'   final_kernels <- as_kernels(tune_res, kernel_mix)
#'   
#'   print("Best configurations found:")
#'   print(final_kernels)
#'
#'   # 4. Train Random Machines (Bagging)
#'   #    Now we use the optimized 'final_kernels' to train the ensemble.
#'   
#'   cat("Training Random Machines ensemble...\n")
#'   rm_model <- random_machines(
#'     data         = train_df,
#'     newdata      = test_df,
#'     time_col     = "tempo",
#'     delta_col    = "cens",
#'     kernels      = final_kernels, # Use tuned kernels
#'     B            = 50,            # Number of bootstrap samples
#'     mtry         = NULL,          # Use all features (Random Forest style)
#'     crop         = 0.10,          # Prune kernels with weight < 10%
#'     prop_holdout = 0.20,          # 20% internal holdout for weighting
#'     cores        = parallel::detectCores(),
#'     seed         = 42,
#'     .progress    = TRUE
#'   )
#'
#'   # 5. Evaluate and Print
#'   print(rm_model)
#'   
#'   cidx <- score(rm_model, test_df)
#'   cat(sprintf("Final Test C-Index: %.4f\n", cidx))
#' }
#' }
#' @seealso \code{\link{tune_random_machines}}, \code{\link{grid_kernel}}, \code{\link{fastsvm}}
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

  # Optional emo package check
  has_emo <- requireNamespace("emo", quietly = TRUE)
  ji <- function(x, fallback = "") if (has_emo) emo::ji(x) else fallback

  if (!is.null(seed)) set.seed(seed)

  n            <- nrow(data)
  kernel_names <- names(kernels)

  # ------------------------------------------------------------------
  # 1. Internal Holdout for Kernel Weights
  # ------------------------------------------------------------------

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

  cli::cli_alert_info("Computing kernel weights...")

  for (i in seq_along(kernel_names)) {
    kname <- kernel_names[i]
    spec  <- kernels[[kname]]

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

    base_cindex[i] <- tryCatch(
      {
        mod <- do.call(fastsvm, args_fit)
        score(mod, d_val_w)
      },
      error = function(e) 0.5
    )
  }

  kernel_lambdas <- .compute_lambdas_from_cindex(base_cindex, beta = beta_kernel)

  if (!is.null(crop)) {
    keep_mask <- kernel_lambdas > crop
    n_kept    <- sum(keep_mask)
    n_total   <- length(kernel_lambdas)

    if (n_kept == 0) {
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

    kernel_lambdas[!keep_mask] <- 0

    if (sum(kernel_lambdas) > 0) {
      kernel_lambdas <- kernel_lambdas / sum(kernel_lambdas)
    }
  }

  py_path_main <- tryCatch(reticulate::py_config()$python, error = function(e) NULL)

  if (cores > 1L) {
    mirai::daemons(n = cores, seed = seed, dispatcher = TRUE)
    on.exit(mirai::daemons(0L), add = TRUE)
  } else if (!is.null(seed)) {
    set.seed(seed)
  }

  # ------------------------------------------------------------------
  # 3. Parallel Bootstrap
  # ------------------------------------------------------------------
  cli::cli_alert_info("Executing parallel bootstrap...")

  boot_results <- purrr::map(
    .x = seq_len(B),
    .f = purrr::in_parallel(
      function(b) {
        if (!requireNamespace("reticulate", quietly = TRUE)) return(NULL)

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

        sksvm_loc  <- reticulate::import("sksurv.svm")
        sksurv_loc <- reticulate::import("sksurv")
        pickle_loc <- reticulate::import("pickle")

        local_mk_surv <- function(t, e) {
          sksurv_loc$util$Surv$from_arrays(
            event = as.logical(e),
            time  = as.numeric(t)
          )
        }

        kname <- sample(kernel_names_ref, size = 1, prob = kernel_lambdas_ref)
        spec  <- kernels_ref[[kname]]

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

        if (!all(x_cols %in% names(newdata_ref))) {
          return(NULL)
        }
        X_test <- as.matrix(newdata_ref[, x_cols, drop = FALSE])

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

        model <- tryCatch({
          mod <- do.call(sksvm_loc$FastKernelSurvivalSVM, final_args)
          mod$fit(X_mat, y_surv)
          mod
        }, error = function(e) NULL)

        if (is.null(model)) {
          return(NULL)
        }

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

        pred_vec <- tryCatch(
          as.numeric(model$predict(X_test)),
          error = function(e) na_pred
        )

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

#' Score method for Random Machines
#' @param object A fitted \code{random_machines} model.
#' @param data Validation data containing time and event columns.
#' @param ... Additional arguments.
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
#' @param x An object of class \code{"random_machines"}.
#' @param ... Additional arguments.
#' @export
print.random_machines <- function(x, ...) {
  if (!requireNamespace("cli", quietly = TRUE)) {
    print.default(x)
    return(invisible(x))
  }

  has_emo <- requireNamespace("emo", quietly = TRUE)
  ji <- function(x, fallback = "") if (has_emo) emo::ji(x) else fallback

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

  # Table 1
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

  w_kern_u  <- max(nchar(df_usage$Kernel), nchar("Kernel"))
  w_count_u <- max(nchar(as.character(df_usage$Count)), nchar("Count"))
  w_prob_u  <- max(nchar("Probability"), nchar("0.0000"))

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

  # Table 2
  cli::cli_h2(paste(ji("balance_scale", "||"), "Kernel Weights (Holdout Probabilities)"))

  df_w <- data.frame(
    Kernel = names(x$kernel_lambdas),
    Prob   = as.numeric(x$kernel_lambdas),
    stringsAsFactors = FALSE
  )
  df_w <- df_w[order(-df_w$Prob), ]

  w_kern_w <- max(nchar(df_w$Kernel), nchar("Kernel"))
  w_prob_w <- max(nchar("Probability"), nchar("0.0000"))

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
      status_icon <- if(has_emo) ji("white_check_mark") else "OK"
      prob_fmt    <- cli::col_green(prob_fmt)
      status_full <- paste(status_icon, cli::style_bold(cli::col_green("Selected")))
    } else {
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

#' Predict method for Random Machines
#' @param object An object of class \code{"random_machines"}.
#' @param newdata Data frame with new observations.
#' @param ... Additional arguments.
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