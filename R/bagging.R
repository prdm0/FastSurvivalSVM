#' Bagging of FastKernelSurvivalSVM with covariate sub-sampling
#'
#' Fit a bagging ensemble of \code{sksurv.svm.FastKernelSurvivalSVM} models,
#' using \code{fast_kernel_surv_svm_fit()} as the base learner.
#'
#' Each base model is fitted on:
#' \itemize{
#'   \item a bootstrap (or subsample) of the rows, and
#'   \item a random subset of the covariates (\code{mtry}), analogous to
#'         Random Forest, but with kernel survival SVMs instead of trees.
#' }
#'
#' The \code{kernels} argument controls how kernels are chosen:
#' \itemize{
#'   \item If a single value is provided (string or function), the same kernel
#'         is used in all base models.
#'   \item If a \code{list()} is provided, each element must be a valid kernel
#'         (a string or an R function compatible with
#'         \code{fast_kernel_surv_svm_fit()}), and for each base model one
#'         kernel is sampled from this list.
#' }
#'
#' Machine-learning friendly design:
#' \itemize{
#'   \item SVM hyperparameters (\code{alpha}, \code{rank_ratio}, etc.) are
#'         explicit, named arguments of this function, suitable for external
#'         tuning (e.g., with \pkg{tidymodels}).
#'   \item Hyperparameters of custom kernels should be encoded inside the
#'         kernel functions themselves (e.g. \code{make_wavelet_kernel(A = 0.5)}).
#'         The bagging function does \emph{not} re-tune kernel hyperparameters
#'         at each bootstrap; it only samples among already parameterised
#'         kernels.
#' }
#'
#' This package does not implement new kernel functions itself. The user is
#' expected to:
#' \itemize{
#'   \item use strings supported by scikit-learn
#'         (e.g. \code{"rbf"}, \code{"poly"}, \code{"sigmoid"},
#'         \code{"laplacian"}, etc.), or
#'   \item pass R functions of the form \code{function(x, z) ...} that return
#'         a scalar kernel value, as required by \code{FastKernelSurvivalSVM}.
#' }
#'
#' Parallelisation:
#' \itemize{
#'   \item If \code{parallel = TRUE}, base models are fitted in parallel using
#'         \pkg{future.apply} (\code{future_lapply}), allowing any backend
#'         supported by \pkg{future} (multicore, multisession, cluster, etc.).
#'   \item The user must set the execution plan before calling this function,
#'         e.g. \code{future::plan(multisession)}.
#' }
#'
#' @param data A \code{data.frame} containing survival time, event indicator,
#'   and covariates.
#' @param time_col Name of the column in \code{data} containing survival times.
#' @param delta_col Name of the column in \code{data} containing the event
#'   indicator (1 = event, 0 = right censoring).
#' @param n_estimators Integer; number of base learners (bootstrap replicates)
#'   in the ensemble.
#' @param mtry Integer; number of covariates to be used in each base learner.
#'   If \code{NULL}, defaults to \code{floor(sqrt(p))}, where \code{p} is the
#'   number of available covariates.
#' @param sample_frac Fraction of rows to sample for each base learner.
#'   If \code{replace = TRUE}, \code{sample_frac = 1} corresponds to standard
#'   bootstrap resampling. If smaller than 1, it performs subsampling.
#' @param replace Logical; if \code{TRUE}, sample rows with replacement
#'   (bootstrap). If \code{FALSE}, sample without replacement.
#' @param kernels Kernel specification:
#'   \itemize{
#'     \item A single value: either a character string (e.g. \code{"rbf"},
#'           \code{"poly"}) or an R function \code{function(x, z) ...}, used in
#'           all base models.
#'     \item A \code{list()} of kernels (each element a string or function);
#'           for each base model, one kernel is sampled from this list.
#'   }
#'   Kernel functions can be \emph{factory functions} that already fix their
#'   hyperparameters (e.g. \code{make_wavelet_kernel(A = 0.5)}).
#' @param kernel_prob Optional numeric vector of sampling probabilities for
#'   kernels when \code{kernels} is a list. Must have the same length as
#'   \code{kernels}. If \code{NULL}, a uniform distribution over kernels is
#'   used.
#' @param alpha Regularisation parameter for the SVM, passed to
#'   \code{FastKernelSurvivalSVM(alpha = ...)}.
#' @param rank_ratio Mixing parameter between ranking and regression objectives
#'   (\code{0 <= rank_ratio <= 1}), forwarded to the Python estimator.
#'   For pure regression on transformed survival times, use \code{rank_ratio = 0}.
#' @param fit_intercept Logical; if \code{TRUE}, include an intercept when there
#'   is a regression component (\code{rank_ratio < 1}).
#' @param parallel Logical; if \code{TRUE}, fit base learners in parallel using
#'   \pkg{future.apply}. The execution plan must be set by the user using
#'   \code{future::plan()}.
#' @param seed Optional integer; seed to be set before bagging. Useful for
#'   reproducibility, especially together with \pkg{future}.
#' @param ... Additional arguments passed directly to
#'   \code{sksurv.svm.FastKernelSurvivalSVM()} via
#'   \code{fast_kernel_surv_svm_fit()}, e.g. \code{gamma}, \code{degree},
#'   \code{coef0}, \code{max_iter}, \code{optimizer}, \code{tol}, etc.
#'
#' @return An object of class \code{"fastsvm_bagging"} with components:
#'   \itemize{
#'     \item \code{models}: list of \code{"fastsvm"} base learners;
#'     \item \code{subspaces}: list of character vectors with the covariate
#'           names used in each base learner;
#'     \item \code{time_col}, \code{delta_col}, \code{x_cols}: metadata;
#'     \item \code{rank_ratio}: ensemble-level \code{rank_ratio};
#'     \item \code{n_estimators}, \code{mtry}, \code{sample_frac},
#'           \code{replace}: ensemble configuration.
#'   }
#'
#' @examples
#' if (reticulate::py_module_available("sksurv")) {
#'
#'   ## ------------------------------------------------------------
#'   ## Example 1: Simple bagging with a fixed RBF kernel (regression mode)
#'   ## ------------------------------------------------------------
#'   set.seed(123)
#'   df <- data_generation(n = 300L, prop_cen = 0.10)
#'
#'   bag_rbf <- fastsvm_bagging_fit(
#'     data         = df,
#'     time_col     = "tempo",
#'     delta_col    = "cens",
#'     n_estimators = 10L,
#'     mtry         = 2L,
#'     kernels      = "rbf",   # scikit-learn RBF kernel
#'     alpha        = 1,
#'     rank_ratio   = 0        # pure regression on transformed times
#'   )
#'
#'   preds_rbf <- predict(bag_rbf, df)
#'   head(preds_rbf)
#'
#'   cindex_rbf <- score_fastsvm_bagging(bag_rbf, df)
#'   cindex_rbf
#'
#'
#'   ## ------------------------------------------------------------
#'   ## Example 2: Bagging with a list of kernels (RBF and polynomial)
#'   ## ------------------------------------------------------------
#'   kernels_list <- list("rbf", "poly")
#'
#'   bag_mix <- fastsvm_bagging_fit(
#'     data         = df,
#'     time_col     = "tempo",
#'     delta_col    = "cens",
#'     n_estimators = 15L,
#'     mtry         = 3L,
#'     kernels      = kernels_list,
#'     kernel_prob  = c(0.7, 0.3), # 70% RBF, 30% polynomial
#'     alpha        = 0.5,
#'     rank_ratio   = 0
#'   )
#'
#'   preds_mix <- predict(bag_mix, df)
#'   head(preds_mix)
#'
#'   cindex_mix <- score_fastsvm_bagging(bag_mix, df)
#'   cindex_mix
#'
#'
#'   ## ------------------------------------------------------------
#'   ## Example 3: Bagging with custom R kernel functions
#'   ##             (including hyperparameters)
#'   ## ------------------------------------------------------------
#'
#'   # Custom wavelet mother function
#'   wavelet_mother <- function(u) {
#'     cos(1.75 * u) * exp(-0.5 * u^2)
#'   }
#'
#'   # Multidimensional wavelet kernel with scale parameter A
#'   wavelet_kernel <- function(x, z, A = 1) {
#'     x <- as.numeric(x)
#'     z <- as.numeric(z)
#'     stopifnot(length(x) == length(z))
#'     stopifnot(length(A) == 1L, A > 0)
#'
#'     u <- (x - z) / A
#'     prod(wavelet_mother(u))
#'   }
#'
#'   # Kernel factory fixing the hyperparameter A
#'   make_wavelet_kernel <- function(A = 1) {
#'     force(A)
#'     function(x, z) wavelet_kernel(x, z, A = A)
#'   }
#'
#'   # Another example: RBF kernel with tunable sigma
#'   make_rbf_kernel <- function(sigma = 1) {
#'     force(sigma)
#'     function(x, z) {
#'       x <- as.numeric(x); z <- as.numeric(z)
#'       d2 <- sum((x - z)^2)
#'       exp(-d2 / (2 * sigma^2))
#'     }
#'   }
#'
#'   custom_kernels <- list(
#'     make_wavelet_kernel(A = 0.5),  # wavelet kernel with A = 0.5
#'     make_rbf_kernel(sigma = 0.8)   # custom RBF kernel with sigma = 0.8
#'   )
#'
#'   bag_custom <- fastsvm_bagging_fit(
#'     data         = df,
#'     time_col     = "tempo",
#'     delta_col    = "cens",
#'     n_estimators = 10L,
#'     mtry         = 2L,
#'     kernels      = custom_kernels,  # list of R kernel functions
#'     alpha        = 1,
#'     rank_ratio   = 0
#'   )
#'
#'   preds_custom <- predict(bag_custom, df)
#'   head(preds_custom)
#'
#'   cindex_custom <- score_fastsvm_bagging(bag_custom, df)
#'   cindex_custom
#'
#'
#'   ## ------------------------------------------------------------
#'   ## Example 4 (optional): Parallel fitting using future/future.apply
#'   ## ------------------------------------------------------------
#'   ## Not run by default on CRAN / examples:
#'   # if (requireNamespace("future.apply", quietly = TRUE)) {
#'   #   future::plan(future::multisession)
#'   #   bag_parallel <- fastsvm_bagging_fit(
#'   #     data         = df,
#'   #     time_col     = "tempo",
#'   #     delta_col    = "cens",
#'   #     n_estimators = 20L,
#'   #     mtry         = 2L,
#'   #     kernels      = "rbf",
#'   #     alpha        = 1,
#'   #     rank_ratio   = 0,
#'   #     parallel     = TRUE,
#'   #     seed         = 999L
#'   #   )
#'   #   score_fastsvm_bagging(bag_parallel, df)
#'   # }
#'
#'
#'   ## ------------------------------------------------------------
#'   ## Example 5: Comparing single FastKernelSurvivalSVM (poly)
#'   ##            vs bagging with randomisation of 3 kernels
#'   ## ------------------------------------------------------------
#'
#'   set.seed(456)
#'   df_poly <- data_generation(n = 300L, prop_cen = 0.10)
#'
#'   ## (a) Single-model FastKernelSurvivalSVM with polynomial kernel
#'   fit_poly <- fast_kernel_surv_svm_fit(
#'     data        = df_poly,
#'     time_col    = "tempo",
#'     delta_col   = "cens",
#'     kernel      = "poly",
#'     alpha       = 2,
#'     rank_ratio  = 0,
#'     degree      = 3L,   # integer required by scikit-learn
#'     coef0       = 1L
#'   )
#'
#'   cindex_poly_single <- score_fastsvm(fit_poly, df_poly)
#'
#'
#'   ## (b) Bagging with 3 kernels: poly, rbf and wavelet
#'
#'   # Wavelet mother function
#'   wavelet_mother2 <- function(u) {
#'     cos(1.75 * u) * exp(-0.5 * u^2)
#'   }
#'
#'   # Wavelet kernel with hyperparameter A
#'   wavelet_kernel2 <- function(x, z, A = 1) {
#'     x <- as.numeric(x); z <- as.numeric(z)
#'     u <- (x - z) / A
#'     prod(wavelet_mother2(u))
#'   }
#'
#'   # Kernel factory: fixes A and returns a function(x, z)
#'   make_wavelet_kernel2 <- function(A = 1) {
#'     force(A)
#'     function(x, z) wavelet_kernel2(x, z, A = A)
#'   }
#'
#'   kernels_3 <- list(
#'     "poly",
#'     "rbf",
#'     make_wavelet_kernel2(A = 0.7)
#'   )
#'
#'   bag_poly_mix <- fastsvm_bagging_fit(
#'     data         = df_poly,
#'     time_col     = "tempo",
#'     delta_col    = "cens",
#'     n_estimators = 20L,
#'     mtry         = 2L,
#'     kernels      = kernels_3,
#'     kernel_prob  = c(0.4, 0.4, 0.2),  # 40% poly, 40% rbf, 20% wavelet
#'     alpha        = 2,
#'     rank_ratio   = 0,
#'     degree       = 3L,                # used only for poly kernel
#'     coef0        = 1L
#'   )
#'
#'   cindex_poly_bagging <- score_fastsvm_bagging(bag_poly_mix, df_poly)
#'
#'   ## Compare C-index: single model vs bagging ensemble
#'   c(
#'     cindex_poly_single  = cindex_poly_single,
#'     cindex_poly_bagging = cindex_poly_bagging
#'   )
#'
#' }
#' @export
fastsvm_bagging_fit <- function(
  data,
  time_col      = "t",
  delta_col     = "delta",
  n_estimators  = 50L,
  mtry          = NULL,
  sample_frac   = 1,
  replace       = TRUE,
  kernels       = "rbf",
  kernel_prob   = NULL,
  alpha         = 1,
  rank_ratio    = 0,
  fit_intercept = FALSE,
  parallel      = FALSE,
  seed          = NULL,
  ...
) {
  stopifnot(time_col  %in% names(data))
  stopifnot(delta_col %in% names(data))

  if (!is.null(seed)) {
    set.seed(seed)
  }

  # available covariates
  x_cols_all <- setdiff(names(data), c(time_col, delta_col))
  p <- length(x_cols_all)
  if (p == 0L) {
    stop("`data` must contain at least one covariate besides time and delta.")
  }

  # default mtry: sqrt(p)
  if (is.null(mtry)) {
    mtry <- max(1L, floor(sqrt(p)))
  }
  if (mtry < 1L || mtry > p) {
    stop("`mtry` must be between 1 and the number of covariates (", p, ").")
  }

  # sample size for each base learner
  n <- nrow(data)
  n_sample <- max(1L, round(sample_frac * n))

  # handle kernel specification
  if (is.list(kernels)) {
    K <- length(kernels)
    if (K == 0L) {
      stop("If `kernels` is a list, it must have at least one element.")
    }
    if (is.null(kernel_prob)) {
      kernel_prob <- rep(1 / K, K)
    } else {
      if (length(kernel_prob) != K) {
        stop("`kernel_prob` must have the same length as `kernels`.")
      }
      if (any(kernel_prob < 0)) {
        stop("`kernel_prob` must have non-negative values.")
      }
      s <- sum(kernel_prob)
      if (s <= 0) {
        stop("The sum of `kernel_prob` must be positive.")
      }
      kernel_prob <- kernel_prob / s
    }
    draw_kernel <- function() {
      idx <- sample.int(K, size = 1L, prob = kernel_prob)
      kernels[[idx]]
    }
  } else {
    # single kernel (string or function)
    draw_kernel <- function() kernels
  }

  # internal function: fit a single base learner
  fit_one_base <- function(b) {
    # 1) row sampling
    idx <- if (replace) {
      sample.int(n, size = n_sample, replace = TRUE)
    } else {
      sample.int(n, size = n_sample, replace = FALSE)
    }
    data_b <- data[idx, , drop = FALSE]

    # 2) covariate sub-sampling
    x_sub <- sample(x_cols_all, size = mtry, replace = FALSE)

    # 3) choose kernel for this base learner
    kernel_b <- draw_kernel()

    # 4) subset data to time, delta, and selected covariates
    data_fit <- data_b[, c(time_col, delta_col, x_sub), drop = FALSE]

    # 5) fit base learner via the existing wrapper
    fit_b <- fast_kernel_surv_svm_fit(
      data          = data_fit,
      time_col      = time_col,
      delta_col     = delta_col,
      kernel        = kernel_b,
      alpha         = alpha,
      rank_ratio    = rank_ratio,
      fit_intercept = fit_intercept,
      ...
    )

    list(
      model    = fit_b,
      subspace = x_sub
    )
  }

  idx_vec <- seq_len(n_estimators)

  if (parallel) {
    if (!requireNamespace("future.apply", quietly = TRUE)) {
      stop("Package `future.apply` is required for parallel = TRUE. ",
           "Please install it or set parallel = FALSE.")
    }
    # user must set the desired future::plan() before calling this function
    res_list <- future.apply::future_lapply(idx_vec, fit_one_base)
  } else {
    res_list <- lapply(idx_vec, fit_one_base)
  }

  models    <- lapply(res_list, function(z) z$model)
  subspaces <- lapply(res_list, function(z) z$subspace)

  structure(
    list(
      models       = models,
      subspaces    = subspaces,
      time_col     = time_col,
      delta_col    = delta_col,
      x_cols       = x_cols_all,
      rank_ratio   = rank_ratio,
      n_estimators = n_estimators,
      mtry         = mtry,
      sample_frac  = sample_frac,
      replace      = replace
    ),
    class = "fastsvm_bagging"
  )
}

#' Predict from a fastsvm_bagging ensemble
#'
#' Compute ensemble predictions for a \code{"fastsvm_bagging"} object as the
#' average of the predictions of its base learners.
#'
#' This is compatible with both ranking (\code{rank_ratio = 1}) and regression
#' (\code{rank_ratio < 1}) modes of \code{FastKernelSurvivalSVM}. For pure
#' regression examples, see \code{\link{fastsvm_bagging_fit}}.
#'
#' @param object An object of class \code{"fastsvm_bagging"} returned by
#'   \code{fastsvm_bagging_fit()}.
#' @param newdata A \code{data.frame} containing the covariates used in
#'   training (must include all columns listed in \code{object$x_cols}).
#' @param ... Additional arguments (unused; included for S3 compatibility).
#'
#' @return A numeric vector of ensemble predictions, one value per row of
#'   \code{newdata}.
#'
#' @examples
#' if (reticulate::py_module_available("sksurv")) {
#'   set.seed(123)
#'   df <- data_generation(n = 200L, prop_cen = 0.2)
#'
#'   bag <- fastsvm_bagging_fit(
#'     data         = df,
#'     time_col     = "tempo",
#'     delta_col    = "cens",
#'     n_estimators = 8,
#'     mtry         = 2,
#'     kernels      = "rbf",
#'     alpha        = 1,
#'     rank_ratio   = 0
#'   )
#'
#'   preds <- predict(bag, df)
#'   head(preds)
#' }
#'
#' @export
predict.fastsvm_bagging <- function(object, newdata, ...) {
  missing_cols <- setdiff(object$x_cols, names(newdata))
  if (length(missing_cols) > 0L) {
    stop("The following covariates are missing from `newdata`: ",
         paste(missing_cols, collapse = ", "))
  }

  n_models <- length(object$models)
  if (n_models == 0L) {
    stop("Object `fastsvm_bagging` has no fitted base models.")
  }

  preds_mat <- sapply(seq_len(n_models), function(b) {
    mdl   <- object$models[[b]]
    x_sub <- object$subspaces[[b]]
    as.numeric(predict(mdl, newdata[, x_sub, drop = FALSE]))
  })

  rowMeans(preds_mat)
}

#' Concordance index for a fastsvm_bagging ensemble
#'
#' Compute the concordance index (C-index) for a \code{"fastsvm_bagging"}
#' model on a given dataset, using the aggregated ensemble predictions.
#'
#' IMPORTANT:
#' \itemize{
#'   \item If \code{rank_ratio = 1}, predictions are **risk scores**
#'         (larger = higher risk → shorter survival).
#'
#'   \item If \code{rank_ratio < 1}, predictions are **transformed survival times**
#'         (larger = longer survival).  
#'         In this case, the predictions are multiplied by \code{-1} before
#'         computing the C-index, because scikit-survival requires
#'         “larger = higher risk”.
#' }
#'
#' Internally, this uses \code{sksurv.metrics.concordance_index_censored}
#' through \pkg{reticulate}.
#'
#' @param object An object of class \code{"fastsvm_bagging"}.
#' @param data A \code{data.frame} containing \code{object$time_col},
#'   \code{object$delta_col}, and all covariates used in training.
#'
#' @return A single numeric value with the estimated concordance index.
#'
#' @examples
#' if (reticulate::py_module_available("sksurv")) {
#'
#'   ## ------------------------------------------------------------
#'   ## Example 1: Bagging with fixed RBF kernel (pure regression)
#'   ## ------------------------------------------------------------
#'   set.seed(123)
#'   df <- data_generation(n = 250, prop_cen = 0.15)
#'
#'   bag_rbf <- fastsvm_bagging_fit(
#'     data         = df,
#'     time_col     = "tempo",
#'     delta_col    = "cens",
#'     n_estimators = 10,
#'     mtry         = 2,
#'     kernels      = "rbf",
#'     alpha        = 1,
#'     rank_ratio   = 0
#'   )
#'
#'   c_rbf <- score_fastsvm_bagging(bag_rbf, df)
#'   c_rbf
#'
#'
#'   ## ------------------------------------------------------------
#'   ## Example 2: Bagging mixing kernels (RBF + user-defined wavelet)
#'   ## ------------------------------------------------------------
#'
#'   # wavelet mother function
#'   wavelet_mother <- function(u) {
#'     cos(1.75 * u) * exp(-0.5 * u^2)
#'   }
#'
#'   # wavelet kernel with parameter A
#'   wavelet_kernel <- function(x, z, A = 1) {
#'     x <- as.numeric(x); z <- as.numeric(z)
#'     u <- (x - z) / A
#'     prod(wavelet_mother(u))
#'   }
#'
#'   # factory for wavelet kernel
#'   make_wavelet_kernel <- function(A = 1) {
#'     force(A)
#'     function(x, z) wavelet_kernel(x, z, A = A)
#'   }
#'
#'   kernels_list <- list(
#'     "rbf",
#'     make_wavelet_kernel(A = 0.5)
#'   )
#'
#'   bag_mix <- fastsvm_bagging_fit(
#'     data         = df,
#'     time_col     = "tempo",
#'     delta_col    = "cens",
#'     n_estimators = 12,
#'     mtry         = 2,
#'     kernels      = kernels_list,
#'     kernel_prob  = c(0.6, 0.4),
#'     alpha        = 1,
#'     rank_ratio   = 0
#'   )
#'
#'   c_mix <- score_fastsvm_bagging(bag_mix, df)
#'   c_mix
#'
#' }
#'
#' @export
score_fastsvm_bagging <- function(object, data) {
  stopifnot(object$time_col  %in% names(data))
  stopifnot(object$delta_col %in% names(data))

  # Ensemble prediction
  preds <- predict(object, data)

  # Convert to risk scores
  risk <- if (object$rank_ratio < 1) {
    -preds    # transformed survival times → risk
  } else {
    preds     # already risk scores
  }

  # Build Surv-like structured array
  event <- data[[object$delta_col]] == 1L
  y     <- .mk_surv_object(time = data[[object$time_col]], event = event)

  # Call scikit-survival C-index
  if (!exists("sksurv")) {
    stop("Python module `sksurv` must be loaded and bound to `sksurv`.")
  }

  cindex_fun <- sksurv$metrics$concordance_index_censored
  res <- cindex_fun(event, data[[object$time_col]], risk)

  as.numeric(res[[1]])
}

#' Print method for fastsvm_bagging objects
#'
#' Display a compact summary of a \code{"fastsvm_bagging"} ensemble, including
#' the number of base learners, number of covariates, and key configuration
#' parameters.
#'
#' @param x An object of class \code{"fastsvm_bagging"}.
#' @param ... Not used.
#'
#' @examples
#' if (reticulate::py_module_available("sksurv")) {
#'   set.seed(123)
#'   df <- data_generation(n = 100L, prop_cen = 0.2)
#'
#'   bag <- fastsvm_bagging_fit(
#'     data         = df,
#'     time_col     = "tempo",
#'     delta_col    = "cens",
#'     n_estimators = 5,
#'     mtry         = 2,
#'     kernels      = "rbf",
#'     alpha        = 1,
#'     rank_ratio   = 0
#'   )
#'
#'   print(bag)
#' }
#'
#' @export
print.fastsvm_bagging <- function(x, ...) {
  n_models <- length(x$models)
  p        <- length(x$x_cols)

  cat("Ensemble FastKernelSurvivalSVM (bagging)\n")
  cat("=========================================\n")
  cat(sprintf(" - n_estimators : %d\n", x$n_estimators))
  cat(sprintf(" - p (covariates): %d\n", p))
  cat(sprintf(" - mtry          : %d\n", x$mtry))
  cat(sprintf(" - sample_frac   : %.3f\n", x$sample_frac))
  cat(sprintf(" - replace       : %s\n", x$replace))
  cat(sprintf(" - rank_ratio    : %.4g\n", x$rank_ratio))
  cat("\nCovariates (total):\n")
  if (p <= 12) {
    cat("  ", paste(x$x_cols, collapse = ", "), "\n")
  } else {
    cat("  ", paste(x$x_cols[1:12], collapse = ", "),
        ", ... (total = ", p, ")\n", sep = "")
  }
  invisible(x)
}
