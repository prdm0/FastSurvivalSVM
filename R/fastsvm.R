# R/fast_kernel_surv_svm.R

#' Build a scikit-survival Surv-like object from R vectors
#'
#' Internal helper that converts time and event vectors into the
#' structured array expected by \code{sksurv.util.Surv.from_arrays()}.
#'
#' @param time Numeric vector of survival times.
#' @param event Logical or numeric vector indicating the event
#'   (TRUE/1 = event, FALSE/0 = right censoring).
#'
#' @keywords internal
.mk_surv_object <- function(time, event) {
  # event: TRUE = event, FALSE = censoring
  Surv <- sksurv$util$Surv
  Surv$from_arrays(
    event = as.logical(event),
    time  = as.numeric(time)
  )
}

#' Fit FastKernelSurvivalSVM (scikit-survival) from an R data frame
#'
#' This function wraps the Python implementation of
#' \code{sksurv.svm.FastKernelSurvivalSVM} and provides a convenient
#' R interface for fitting kernel-based survival SVMs to
#' right-censored data.
#'
#' The input \code{data} must contain a time column, an event indicator
#' column, and one or more covariate columns. Internally, the function
#' constructs the survival outcome in the format required by
#' scikit-survival and calls the Python estimator via \pkg{reticulate}.
#'
#' @param data A \code{data.frame} with survival times, event indicator,
#'   and covariates.
#' @param time_col Name of the column in \code{data} containing survival times.
#' @param delta_col Name of the column in \code{data} containing the event
#'   indicator (1 = event, 0 = right censoring).
#' @param kernel Either a character string specifying a kernel supported by
#'   scikit-learn (for example \code{"rbf"}, \code{"poly"}, \code{"sigmoid"}),
#'   or an R function of the form \code{function(x, z) ...} that takes two
#'   numeric vectors and returns a scalar kernel value.
#' @param alpha Regularization parameter controlling the weight of the squared
#'   hinge loss in the objective function (see scikit-survival documentation).
#' @param rank_ratio Mixing parameter between regression and ranking objectives,
#'   with \code{0 <= rank_ratio <= 1}. Use \code{0} for pure regression and
#'   \code{1} for pure ranking.
#' @param fit_intercept Logical; if \code{TRUE}, an intercept is included in
#'   the regression objective (only relevant when \code{rank_ratio < 1}).
#' @param ... Additional arguments passed directly to
#'   \code{sksurv.svm.FastKernelSurvivalSVM()}.
#'
#' @return An object of class \code{"fastsvm"}, which wraps the underlying
#'   Python model and stores meta-information about the fit.
#'
#' @examples
#' if (reticulate::py_module_available("sksurv")) {
#'   set.seed(1)
#'   n <- 100
#'   df <- data.frame(
#'     time   = rexp(n, rate = 0.1),
#'     status = rbinom(n, 1, 0.7),  # 1 = event, 0 = censoring
#'     x1     = rnorm(n),
#'     x2     = rnorm(n)
#'   )
#'
#'   # Example 1: using a built-in RBF kernel from scikit-learn
#'   fit_rbf <- fastsvm(
#'     data        = df,
#'     time_col    = "time",
#'     delta_col   = "status",
#'     kernel      = "rbf",
#'     alpha       = 1,
#'     rank_ratio  = 0   # pure regression
#'   )
#'
#'   # Predictions (transformed survival times / risk scores)
#'   y_hat <- predict(fit_rbf, df)
#'   head(y_hat)
#'
#'   # Concordance index on the training data
#'   cidx <- score(fit_rbf, df)
#'   cidx
#'
#'   # Example 2: using a custom RBF kernel defined in R
#'   rbf_r <- function(x, z, sigma = 1) {
#'     d2 <- sum((x - z)^2)
#'     exp(-d2 / (2 * sigma^2))
#'   }
#'
#'   fit_custom <- fastsvm(
#'     data        = df,
#'     time_col    = "time",
#'     delta_col   = "status",
#'     kernel      = function(x, z) rbf_r(x, z, sigma = 0.5),
#'     alpha       = 1,
#'     rank_ratio  = 0
#'   )
#'
#'   summary(fit_custom)
#' }
#'
#' @export
fastsvm <- function(
  data,
  time_col   = "t",
  delta_col  = "delta",
  kernel     = "rbf",
  alpha      = 1,
  rank_ratio = 0,
  fit_intercept = TRUE,
  ...
) {
  stopifnot(time_col  %in% names(data))
  stopifnot(delta_col %in% names(data))
  
  x_cols <- setdiff(names(data), c(time_col, delta_col))
  if (length(x_cols) == 0L) {
    stop("At least one covariate column (besides time and delta) is required.")
  }
  
  X <- as.matrix(data[, x_cols, drop = FALSE])
  
  # Assuming delta = 1 event, 0 censoring:
  event <- data[[delta_col]] == 1L
  y     <- .mk_surv_object(time = data[[time_col]], event = event)
  
  # Bridge kernel R -> Python
  kernel_py <- if (is.character(kernel)) {
    kernel
  } else if (is.function(kernel)) {
    # FastKernelSurvivalSVM expects a callable (x_row, z_row) -> scalar
    function(x_row, z_row) {
      x_r <- as.numeric(x_row)
      z_r <- as.numeric(z_row)
      kernel(x_r, z_r)
    }
  } else {
    stop("`kernel` must be either a character string or an R function.")
  }
  
  model <- sksvm$FastKernelSurvivalSVM(
    alpha         = alpha,
    rank_ratio    = rank_ratio,
    fit_intercept = fit_intercept,
    kernel        = kernel_py,
    ...
  )
  
  model$fit(X, y)
  
  structure(
    list(
      model       = model,
      time_col    = time_col,
      delta_col   = delta_col,
      x_cols      = x_cols,
      rank_ratio  = rank_ratio
    ),
    class = "fastsvm"
  )
}

#' Print method for fastsvm objects
#'
#' This method prints a compact summary of the fitted
#' \code{"fastsvm"} model, including the number of observations,
#' number of covariates, kernel type, and key hyperparameters.
#'
#' @param x An object of class \code{"fastsvm"}.
#' @param ... Not used.
#'
#' @export
print.fastsvm <- function(x, ...) {
  model <- x$model
  
  cat("FastKernelSurvivalSVM model (Python via scikit-survival)\n")
  cat("--------------------------------------------------------\n")
  
  # Number of training observations
  n_obs <- length(model$coef_)
  cat(sprintf("Data:\n  - n (observations): %d\n  - p (covariates)  : %d\n",
              n_obs, length(x$x_cols)))
  
  if (length(x$x_cols) > 0L) {
    if (length(x$x_cols) <= 12) {
      cat("  - Covariates     : ", paste(x$x_cols, collapse = ", "), "\n", sep = "")
    } else {
      cat("  - Covariates     : ",
          paste(x$x_cols[1:12], collapse = ", "),
          ", ... (total = ", length(x$x_cols), ")\n", sep = "")
    }
  }
  
  # Kernel
  ker_r <- tryCatch(reticulate::py_to_r(model$get_params()[["kernel"]]),
                    error = function(e) NULL)
  cat("\nKernel:\n")
  if (is.character(ker_r)) {
    cat(sprintf("  - type           : '%s' (scikit-learn built-in)\n", ker_r))
  } else {
    cat("  - type           : custom callable function\n")
  }
  
  # Hyperparameters
  alpha_val  <- as.numeric(model$get_params()[["alpha"]])
  fit_int    <- as.logical(model$get_params()[["fit_intercept"]])
  cat("\nHyperparameters:\n")
  cat(sprintf("  - alpha          : %.4g\n", alpha_val))
  cat(sprintf("  - rank_ratio     : %.4g (0 = pure regression)\n", x$rank_ratio))
  cat(sprintf("  - fit_intercept  : %s\n", fit_int))
  
  # Iterations (if available)
  if (!is.null(model$n_iter_)) {
    cat(sprintf("\nOptimization:\n  - iterations     : %d\n", model$n_iter_))
  }
  
  invisible(x)
}

#' Predict risk scores or transformed survival times from a fastsvm model
#'
#' @param object An object of class \code{"fastsvm"} returned by
#'   \code{fastsvm()}.
#' @param newdata A \code{data.frame} containing the covariate columns used
#'   in the original fit (same names as in \code{object$x_cols}).
#' @param ... Not used (for S3 compatibility).
#'
#' @return A numeric vector of predictions:
#'   \itemize{
#'     \item If \code{rank_ratio = 1}, higher values indicate higher risk.
#'     \item If \code{rank_ratio < 1}, values correspond to transformed
#'           survival times (lower = shorter survival, higher = longer).
#'   }
#'
#' @examples
#' if (reticulate::py_module_available("sksurv")) {
#'   set.seed(1)
#'   n <- 50
#'   df <- data.frame(
#'     time   = rexp(n, 0.1),
#'     status = rbinom(n, 1, 0.6),
#'     x1     = rnorm(n),
#'     x2     = rnorm(n)
#'   )
#'
#'   fit <- fastsvm(
#'     data      = df,
#'     time_col  = "time",
#'     delta_col = "status",
#'     kernel    = "rbf"
#'   )
#'
#'   preds <- predict(fit, df)
#'   head(preds)
#' }
#'
#' @export
predict.fastsvm <- function(object, newdata, ...) {
  Xnew <- as.matrix(newdata[, object$x_cols, drop = FALSE])
  as.numeric(object$model$predict(Xnew))
}

#' Concordance index for a fastsvm model
#'
#' Computes the concordance index (C-index) for a fitted \code{"fastsvm"} model
#' on a given data set. This is a convenience wrapper around the Python
#' \code{score()} method.
#'
#' @param object An object of class \code{"fastsvm"}.
#' @param data A \code{data.frame} in the same format used for fitting the
#'   model (must contain \code{object$time_col}, \code{object$delta_col}
#'   and all covariates in \code{object$x_cols}).
#' @param ... Not used (for S3 compatibility).
#'
#' @return A single numeric value with the estimated concordance index.
#'
#' @examples
#' if (reticulate::py_module_available("sksurv")) {
#'   set.seed(123)
#'   n <- 80
#'   df <- data.frame(
#'     time   = rexp(n, 0.15),
#'     status = rbinom(n, 1, 0.65),
#'     x1     = rnorm(n),
#'     x2     = rnorm(n)
#'   )
#'
#'   fit <- fastsvm(
#'     data      = df,
#'     time_col  = "time",
#'     delta_col = "status",
#'     kernel    = "rbf"
#'   )
#'
#'   score(fit, df)
#' }
#'
#' @export
score.fastsvm <- function(object, data, ...) {
  X <- as.matrix(data[, object$x_cols, drop = FALSE])
  
  event <- data[[object$delta_col]] == 1L
  y     <- .mk_surv_object(time = data[[object$time_col]], event = event)
  
  as.numeric(object$model$score(X, y))
}

#' Summary method for fastsvm objects
#'
#' Extracts key information from a fitted \code{"fastsvm"} model, including
#' the number of observations, number of covariates, kernel specification,
#' hyperparameters, and a summary of the sample-wise coefficients
#' \code{coef_} returned by the Python model.
#'
#' @param object An object of class \code{"fastsvm"}.
#' @param ... Not used.
#'
#' @return An object of class \code{"summary.fastsvm"}.
#'
#' @export
summary.fastsvm <- function(object, ...) {
  model <- object$model
  
  coefs  <- as.numeric(model$coef_)
  sv_tol <- 1e-8
  sv     <- abs(coefs) > sv_tol
  
  res <- list(
    n             = length(coefs),
    p             = length(object$x_cols),
    x_cols        = object$x_cols,
    kernel        = model$get_params()[["kernel"]],
    alpha         = as.numeric(model$get_params()[["alpha"]]),
    rank_ratio    = object$rank_ratio,
    fit_intercept = as.logical(model$get_params()[["fit_intercept"]]),
    coef          = coefs,
    n_sv          = sum(sv),
    coef_sum      = summary(coefs),
    n_iter        = if (!is.null(model$n_iter_)) model$n_iter_ else NA_integer_
  )
  
  class(res) <- "summary.fastsvm"
  res
}

#' Print method for summary.fastsvm objects
#'
#' @param x An object of class \code{"summary.fastsvm"}.
#' @param ... Not used.
#'
#' @export
print.summary.fastsvm <- function(x, ...) {
  cat("Summary of FastKernelSurvivalSVM model (kernel survival SVM)\n")
  cat("======================================================================\n\n")
  
  cat("== Data ==\n")
  cat(sprintf("- n (observations) : %d\n", x$n))
  cat(sprintf("- p (covariates)   : %d\n", x$p))
  if (x$p <= 12) {
    cat("- Covariates       : ", paste(x$x_cols, collapse = ", "), "\n\n")
  } else {
    cat("- Covariates       : ",
        paste(x$x_cols[1:12], collapse = ", "),
        ", ... (total = ", x$p, ")\n\n", sep = "")
  }
  
  cat("== Hyperparameters ==\n")
  ker_r <- tryCatch(reticulate::py_to_r(x$kernel), error = function(e) NULL)
  if (is.character(ker_r)) {
    cat(sprintf("- kernel           : '%s'\n", ker_r))
  } else {
    cat("- kernel           : custom callable function\n")
  }
  cat(sprintf("- alpha            : %.4g\n", x$alpha))
  cat(sprintf("- rank_ratio       : %.4g (0 = pure regression)\n", x$rank_ratio))
  cat(sprintf("- fit_intercept    : %s\n\n", x$fit_intercept))
  
  cat("== Estimated parameters (coef_ = sample-wise weights alpha_i) ==\n")
  cat(sprintf("- Number of support-like vectors (|alpha_i| > 1e-8): %d\n", x$n_sv))
  cat("- Summary of alpha_i (coef_):\n")
  print(x$coef_sum)
  
  if (!is.na(x$n_iter)) {
    cat(sprintf("\n- Number of optimization iterations: %d\n", x$n_iter))
  }
  
  cat("======================================================================\n")
  invisible(x)
}

#' Extract sample-wise coefficients (alpha_i) from a fastsvm model
#'
#' This method returns the vector of coefficients \code{coef_} estimated by
#' \code{sksurv.svm.FastKernelSurvivalSVM}. Each coefficient corresponds to
#' the weight assigned to an individual training sample in the kernel-induced
#' decision function.
#'
#' These coefficients play the same role as support vector weights in
#' classical SVMs: samples with non-zero coefficients (within a tolerance)
#' can be interpreted as "support-like" vectors.
#'
#' @param object An object of class \code{"fastsvm"} returned by
#'   \code{fastsvm()}.
#' @param ... Additional arguments (unused; included for S3 compatibility).
#'
#' @return A numeric vector of length \code{n}, containing the sample-wise
#'   coefficients \code{alpha_i}.
#'
#' @examples
#' if (reticulate::py_module_available("sksurv")) {
#'   set.seed(1)
#'   df <- data.frame(
#'     time = rexp(50, 0.1),
#'     status = rbinom(50, 1, 0.7),
#'     x1 = rnorm(50),
#'     x2 = rnorm(50)
#'   )
#'
#'   fit <- fastsvm(
#'     data = df,
#'     time_col = "time",
#'     delta_col = "status",
#'     kernel = "rbf"
#'   )
#'
#'   coef(fit)  # extract coefficients
#' }
#'
#' @export
coef.fastsvm <- function(object, ...) {
  as.numeric(object$model$coef_)
}

#' Extract hyperparameters of a fastsvm model
#'
#' This function retrieves all hyperparameters used in the underlying Python
#' model \code{sksurv.svm.FastKernelSurvivalSVM} by calling its
#' \code{get_params()} method. The returned list contains all estimator
#' arguments, including kernel specification, regularization strength,
#' optimizer settings, and kernel parameters (e.g., \code{gamma}, \code{degree},
#' \code{coef0}).
#'
#' This can be useful for reproducibility, model inspection, or when saving
#' configurations for automated tuning workflows.
#'
#' @param object An object of class \code{"fastsvm"}.
#' @param ... Additional arguments (unused; included for S3 method compatibility).
#'
#' @return A named list of hyperparameters corresponding to those returned by
#'   the Python method \code{FastKernelSurvivalSVM.get_params()}.
#'
#' @examples
#' if (reticulate::py_module_available("sksurv")) {
#'   set.seed(123)
#'
#'   df <- data.frame(
#'     time = rexp(40, 0.2),
#'     status = rbinom(40, 1, 0.6),
#'     x1 = rnorm(40),
#'     x2 = rnorm(40)
#'   )
#'
#'   fit <- fastsvm(
#'     data = df,
#'     time_col = "time",
#'     delta_col = "status",
#'     kernel = "rbf",
#'     alpha = 0.5,
#'     rank_ratio = 0.3
#'   )
#'
#'   get_params_fastsvm(fit)
#' }
#'
#' @export
get_params_fastsvm <- function(object, ...) {
  reticulate::py_to_r(object$model$get_params())
}
