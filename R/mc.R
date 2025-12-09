# R/mc.R

# -------------------------------------------------------------------
# Internal helper for emojis (via 'emo', if available)
# -------------------------------------------------------------------

.mc_emoji <- function(name, fallback) {
  if (requireNamespace("emo", quietly = TRUE)) {
    out <- tryCatch(emo::ji(name), error = function(e) fallback)
    if (is.character(out) && length(out) == 1L) {
      return(out)
    } else {
      return(fallback)
    }
  }
  fallback
}

# -------------------------------------------------------------------
# Internal helper to silence Python warnings via reticulate
# -------------------------------------------------------------------

.mc_silence_py_warnings <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    return(invisible(NULL))
  }
  available <- tryCatch(
    reticulate::py_available(initialize = TRUE),
    error = function(e) FALSE
  )
  if (!available) return(invisible(NULL))

  cmd <- paste(
    "import warnings",
    "warnings.filterwarnings('ignore')",
    sep = "\n"
  )
  try(reticulate::py_run_string(cmd), silent = TRUE)
  invisible(NULL)
}

# -------------------------------------------------------------------
# Monte Carlo Simulation: Bagging vs Individual Kernels
# -------------------------------------------------------------------

#' Monte Carlo Simulation: Bagging vs Individual Kernels
#'
#' This function runs a Monte Carlo simulation comparing individual
#' \code{FastKernelSurvivalSVM} models (one per kernel specification)
#' against a Bagging ensemble fitted via \code{\link{fastsvm_bagging}}.
#'
#' For each replication, the function:
#' \enumerate{
#'   \item Generates a dataset via \code{data_generation()}.
#'   \item Splits it into training and test sets.
#'   \item Fits one \code{\link{fastsvm}} model for each kernel in \code{kernels}
#'         and computes the test C-index via \code{\link{score}}.
#'   \item Fits a Bagging ensemble with \code{\link{fastsvm_bagging}} and
#'         computes its test C-index via \code{\link{score}}.
#' }
#'
#' Bagging is always fitted in parallel according to the \code{cores} argument,
#' using the internal parallelization provided by \code{\link{fastsvm_bagging}}.
#' The Monte Carlo loop itself runs serially, but a global progress bar
#' can be shown via \code{.progress = TRUE}.
#'
#' The console output is slightly "embellished" with emojis (via the
#' \pkg{emo} package, if available) to make the simulation more informative
#' and visually pleasant.
#'
#' @param n Integer. Sample size per replication.
#' @param prop_cen Numeric. Proportion of censoring (0 to 1). Default is 0.3.
#' @param kernels A named list of kernel specifications. The same list is used
#'   for individual fits and for the Bagging ensemble.
#' @param B Integer. Number of bootstrap samples for the Bagging model.
#' @param mtry Integer or \code{NULL}. Number of variables to randomly sample
#'   at each split in the Bagging model (Random Subspace).
#' @param seed Integer or \code{NULL}. Global seed for reproducibility.
#'   This seed is set once at the beginning of the simulation.
#' @param cores Integer. Number of cores used by the Bagging model.
#'   This value is passed to \code{\link{fastsvm_bagging}}.
#' @param train_prop Numeric. Proportion of data used for training (default 0.7).
#'   The remaining data is used for testing.
#' @param n_rep Integer. Number of Monte Carlo replications.
#' @param .progress Logical. If \code{TRUE}, show a progress bar during
#'   Monte Carlo replications (delegated to \code{purrr::map()}).
#'
#' @return A \code{data.frame} summarizing the C-index distribution for each
#'   model across replications, with columns:
#'   \itemize{
#'     \item \code{Model}: kernel name or "Bagging".
#'     \item \code{Mean_C_Index}: mean C-index over replications.
#'     \item \code{SD_C_Index}: standard deviation of the C-index.
#'     \item \code{Min_C_Index}: minimum C-index.
#'     \item \code{Max_C_Index}: maximum C-index.
#'   }
#'   The object also carries an attribute \code{"cindex_matrix"} which is a
#'   matrix of dimension \code{n_rep x n_models} containing the raw C-index
#'   values for each replication (rows) and model (columns).
#'
#' @examples
#' \dontrun{
#' if (reticulate::py_module_available("sksurv") &&
#'     requireNamespace("mirai", quietly = TRUE) &&
#'     requireNamespace("purrr", quietly = TRUE)) {
#'   library(FastSurvivalSVM)
#'
#'   # 1. Custom Kernel Factories
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
#'   # 2. Kernel Specifications (rank_ratio = 0 for Regression / Time)
#'   kernel_mix <- list(
#'     linear   = list(kernel = "linear",
#'                     alpha  = 1,
#'                     rank_ratio = 0,
#'                     fit_intercept = TRUE),
#'     rbf      = list(kernel = "rbf",
#'                     alpha  = 0.5,
#'                     gamma  = 0.1,
#'                     rank_ratio = 0,
#'                     fit_intercept = TRUE),
#'     poly_std = list(kernel = "poly",
#'                     degree = 2L,
#'                     alpha  = 1,
#'                     rank_ratio = 0,
#'                     fit_intercept = TRUE),
#'     wavelet  = list(kernel = make_wavelet(A = 1),
#'                     alpha  = 1,
#'                     rank_ratio = 0,
#'                     fit_intercept = TRUE),
#'     poly_fun = list(kernel = make_poly(degree = 2L),
#'                     alpha  = 1,
#'                     rank_ratio = 0,
#'                     fit_intercept = TRUE)
#'   )
#'
#'   # 3. Run Monte Carlo Simulation (e.g. 20 replications)
#'   results <- mc(
#'     n           = 300,
#'     prop_cen    = 0.3,
#'     kernels     = kernel_mix,
#'     B           = 50,
#'     mtry        = NULL,
#'     seed        = 5,
#'     cores       = parallel::detectCores(),
#'     train_prop  = 0.7,
#'     n_rep       = 20,
#'     .progress   = TRUE
#'   )
#'
#'   print(results)
#' }
#' }
#'
#' @importFrom caret createDataPartition
#' @export
mc <- function(
  n           = 300,
  prop_cen    = 0.3,
  kernels,
  B           = 50,
  mtry        = NULL,
  seed        = NULL,
  cores       = parallel::detectCores(),
  train_prop  = 0.7,
  n_rep       = 50L,
  .progress   = TRUE
) {
  if (!is.null(seed)) set.seed(seed)

  if (!requireNamespace("caret", quietly = TRUE))
    stop("Package 'caret' needed. Please install it.")
  if (!requireNamespace("purrr", quietly = TRUE))
    stop("Package 'purrr' needed. Please install it.")

  # Silence Python warnings globally for this session (if possible)
  .mc_silence_py_warnings()

  kernel_names <- names(kernels)
  if (is.null(kernel_names) || length(kernel_names) == 0L)
    stop("`kernels` must be a named list with at least one element.")

  rocket      <- .mc_emoji("rocket",   "🚀")
  gears       <- .mc_emoji("gear",     "⚙️")
  target      <- .mc_emoji("dart",     "🎯")
  chart       <- .mc_emoji("chart",    "📈")
  loop_emoji  <- .mc_emoji("repeat",   "🔁")
  indiv_emoji <- .mc_emoji("mag",      "🔍")
  bag_emoji   <- .mc_emoji("package",  "📦")

  message(sprintf(
    "%s Monte Carlo simulation started (n_rep = %d, n = %d, prop_cen = %.2f)",
    rocket, n_rep, n, prop_cen
  ))
  message(sprintf(
    "%s Models considered: %s",
    gears, paste(c(kernel_names, "Bagging"), collapse = ", ")
  ))
  message(sprintf(
    "%s Training proportion: %.2f | Bagging cores: %d",
    target, train_prop, cores
  ))

  # ------------------------------------------------------------------
  # Helper: one Monte Carlo replication (serial)
  # ------------------------------------------------------------------
  simulate_once <- function(rep_id) {
    message(sprintf(
      "%s [MC %d/%d] Fitting individual models...",
      indiv_emoji, rep_id, n_rep
    ))

    # 1. Generate data
    df <- data_generation(n = n, prop_cen = prop_cen)

    # 2. Train / Test split (stratified on censoring indicator)
    train_idx <- caret::createDataPartition(df$cens, p = train_prop, list = FALSE)
    train_df  <- df[train_idx, ]
    test_df   <- df[-train_idx, ]

    # 3. Fit individual models
    cidx_vec <- numeric(length(kernel_names) + 1L)
    names(cidx_vec) <- c(kernel_names, "Bagging")

    for (kname in kernel_names) {
      spec <- kernels[[kname]]

      # Defaults consistent with Bagging
      if (is.null(spec$rank_ratio))    spec$rank_ratio    <- 0
      if (is.null(spec$fit_intercept)) spec$fit_intercept <- TRUE

      args_fit <- c(
        list(
          data      = train_df,
          time_col  = "tempo",
          delta_col = "cens"
        ),
        spec
      )

      c_val <- NA_real_

      tryCatch({
        mod   <- do.call(fastsvm, args_fit)
        c_val <- score(mod, test_df)
      }, error = function(e) {
        warning(sprintf(
          "[MC %d] Individual fit for '%s' failed: %s",
          rep_id, kname, e$message
        ))
        c_val <- NA_real_
      })

      cidx_vec[kname] <- c_val
    }

    # 4. Bagging (always parallel internally, using `cores`)
    message(sprintf(
      "%s [MC %d/%d] Fitting Bagging ensemble...",
      bag_emoji, rep_id, n_rep
    ))

    bag_c <- NA_real_

    tryCatch({
      bag_mod <- fastsvm_bagging(
        data      = train_df,
        time_col  = "tempo",
        delta_col = "cens",
        kernels   = kernels,
        B         = B,
        mtry      = mtry,
        cores     = cores,
        .progress = FALSE  # no progress bar in bagging, only at MC level
      )
      bag_c <- score(bag_mod, test_df)
    }, error = function(e) {
      warning(sprintf("[MC %d] Bagging execution failed: %s", rep_id, e$message))
      bag_c <- NA_real_
    })

    cidx_vec["Bagging"] <- bag_c

    cidx_vec
  }

  # ------------------------------------------------------------------
  # Monte Carlo (serial, with progress bar via purrr)
  # ------------------------------------------------------------------
  mc_list <- purrr::map(
    .x = seq_len(n_rep),
    .f = \(rep_id) {
      message(sprintf(
        "%s [MC %d/%d] Starting replication...",
        loop_emoji, rep_id, n_rep
      ))
      simulate_once(rep_id)
    },
    .progress = .progress
  )

  # ------------------------------------------------------------------
  # Aggregation of Monte Carlo results
  # ------------------------------------------------------------------
  cindex_mat <- do.call(rbind, mc_list)
  colnames(cindex_mat) <- names(mc_list[[1]])
  rownames(cindex_mat) <- paste0("rep", seq_len(n_rep))

  mean_c <- colMeans(cindex_mat, na.rm = TRUE)
  sd_c   <- apply(cindex_mat, 2, stats::sd,   na.rm = TRUE)
  min_c  <- apply(cindex_mat, 2, min,        na.rm = TRUE)
  max_c  <- apply(cindex_mat, 2, max,        na.rm = TRUE)

  results <- data.frame(
    Model        = names(mean_c),
    Mean_C_Index = as.numeric(mean_c),
    SD_C_Index   = as.numeric(sd_c),
    Min_C_Index  = as.numeric(min_c),
    Max_C_Index  = as.numeric(max_c),
    stringsAsFactors = FALSE
  )

  # Order by mean performance (descending)
  results <- results[order(results$Mean_C_Index, decreasing = TRUE), ]
  rownames(results) <- NULL

  # Store full matrix as attribute
  attr(results, "cindex_matrix") <- cindex_mat

  message(sprintf(
    "%s Monte Carlo simulation finished! Summary of C-index by model:",
    chart
  ))

  results
}
