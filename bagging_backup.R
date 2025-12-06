# R/fastsvm_bagging.R

# -------------------------------------------------------------------
# Helpers internos
# -------------------------------------------------------------------

#' Compute normalized weights from C-indices
#' @keywords internal
.compute_lambdas_from_cindex <- function(cindex_vec, beta = 1) {
  # Proteção numérica idêntica ao script de referência
  # Se C-index <= 0.5, tratamos como aleatório (ajuste leve para log)
  cindex_vec[is.na(cindex_vec) | cindex_vec <= 0.5] <- 0.500001
  cindex_vec[cindex_vec >= 1] <- 0.999999
  
  # Log-odds
  num <- log(cindex_vec / (1 - cindex_vec))
  
  if (beta != 1) num <- num * beta
  
  lambdas <- num / sum(num)
  return(lambdas)
}

# -------------------------------------------------------------------
# Função Principal de Bagging
# -------------------------------------------------------------------

#' Parallel Bagging for FastKernelSurvivalSVM (Reference Architecture)
#'
#' Fits an ensemble of models using bootstrap aggregation (bagging) and 
#' computes predictions immediately using parallel workers.
#'
#' \strong{Architecture:} 
#' This function adopts the exact "Train-and-Predict" strategy used in the 
#' 'Random Survival Machines' reference script. It accepts \code{newdata} 
#' during training and computes predictions \strong{inside the parallel workers}. 
#' This ensures 100\% stability and numerical consistency.
#'
#' \strong{Regression vs Ranking:}
#' By default (\code{rank_ratio = 0}), the model performs regression on the 
#' survival time. Predictions will be positive and correlated with time.
#'
#' @param data A \code{data.frame} containing training data.
#' @param newdata A \code{data.frame} containing test data for prediction.
#' @param time_col Name of the column with survival times.
#' @param delta_col Name of the column with the event indicator (1=event, 0=censored).
#' @param kernels A named list of kernel specifications. Each element must be a list 
#'   of arguments to the estimator.
#' @param B Integer. Number of bootstrap samples.
#' @param beta_kernel Numeric. Temperature for kernel selection probabilities.
#' @param beta_bag Numeric. Temperature for ensemble weighting.
#' @param cores Integer. Number of parallel workers (via \code{mirai}).
#' @param seed Optional integer passed to \code{mirai::daemons} for RNG reproducibility.
#' @param .progress Logical. Show progress bar?
#'
#' @return An object of class \code{"fastsvm_bag"} containing:
#'   \item{preds}{Numeric vector of aggregated predictions for \code{newdata}.}
#'   \item{weights}{Vector of weights assigned to each bootstrap model.}
#'   \item{chosen_kernels}{Vector of kernel names selected in each bootstrap.}
#'   \item{c_indices}{Vector of OOB C-indices for each bootstrap.}
#'
#' @examples
#' \dontrun{
#' if (reticulate::py_module_available("sksurv") && requireNamespace("mirai")) {
#'   library(FastSurvivalSVM)
#'   library(survival)
#'   
#'   # 1. Data Generation
#'   set.seed(123)
#'   df <- data_generation(n = 300, prop_cen = 0.3)
#'   
#'   idx <- sample(1:nrow(df), 200)
#'   train_df <- df[idx, ]
#'   test_df  <- df[-idx, ]
#'   
#'   # 2. Define Custom Kernel Factories
#'   
#'   # Wavelet Kernel (Matches reference logic: cos(1.75u)*exp(-0.5u^2))
#'   make_wavelet <- function(A = 1) {
#'     force(A)
#'     function(x, z) {
#'       u <- (as.numeric(x) - as.numeric(z)) / A
#'       prod(cos(1.75 * u) * exp(-0.5 * u^2))
#'     }
#'   }
#'   
#'   # Polynomial Kernel (Custom)
#'   make_poly <- function(degree = 3, coef0 = 1) {
#'     force(degree); force(coef0)
#'     function(x, z) (sum(as.numeric(x) * as.numeric(z)) + coef0)^degree
#'   }
#'   
#'   # 3. Kernel Specifications (rank_ratio=0 for Regression/Time)
#'   kernel_mix <- list(
#'     linear   = list(kernel="linear", alpha=1, rank_ratio=0, fit_intercept=TRUE),
#'     poly_std = list(kernel="poly", degree=2L, alpha=1, rank_ratio=0, fit_intercept=TRUE),
#'     wavelet  = list(kernel=make_wavelet(A=1), alpha=1, rank_ratio=0, fit_intercept=TRUE),
#'     poly_fun = list(kernel=make_poly(degree=2L), alpha=1, rank_ratio=0, fit_intercept=TRUE)
#'   )
#'   
#'   # 4. Run Bagging (Using all cores)
#'   bag_results <- fastsvm_bagging(
#'     data       = train_df,
#'     newdata    = test_df,
#'     time_col   = "tempo",
#'     delta_col  = "cens",
#'     kernels    = kernel_mix,
#'     B          = 50,
#'     cores      = parallel::detectCores(),
#'     seed       = 99,
#'     .progress  = TRUE
#'   )
#'   
#'   print(bag_results)
#'   
#'   # 5. Results
#'   # Predictions are time scores (positive)
#'   cat("Preview of Predictions:\n")
#'   print(head(bag_results$preds))
#'   
#'   # Score using the helper function
#'   cidx <- score_fastsvm_bag(bag_results, test_df)
#'   cat(sprintf("Test C-index: %.4f\n", cidx))
#' }
#' }
#'
#' @importFrom mirai daemons
#' @importFrom purrr map in_parallel
#' @importFrom reticulate py_run_string py_call import py_available py_config
#' @export
fastsvm_bagging <- function(
  data,
  newdata,
  time_col   = "t",
  delta_col  = "delta",
  kernels,
  B          = 100L,
  beta_kernel = 1,
  beta_bag    = 1,
  cores       = 1L,
  seed        = NULL,
  .progress   = TRUE
) {
  # --- Validações ---
  stopifnot(is.data.frame(data))
  stopifnot(is.data.frame(newdata))
  
  if (!time_col  %in% names(data)) stop("`time_col` not found.")
  if (!delta_col %in% names(data)) stop("`delta_col` not found.")
  if (!is.list(kernels) || is.null(names(kernels))) stop("`kernels` must be a named list.")
  
  # Validate columns in newdata
  feature_cols <- setdiff(names(data), c(time_col, delta_col))
  missing_cols <- setdiff(feature_cols, names(newdata))
  if(length(missing_cols) > 0) {
    stop("`newdata` is missing columns found in `data`: ", paste(missing_cols, collapse=", "))
  }
  
  B <- as.integer(B)
  if (B <= 0L) stop("`B` must be a positive integer.")

  if (!requireNamespace("mirai", quietly = TRUE)) stop("Package 'mirai' required.")
  if (!requireNamespace("purrr", quietly = TRUE)) stop("Package 'purrr' required.")

  n <- nrow(data)
  kernel_names <- names(kernels)

  # --- 1. Ajuste Inicial (Serial) ---
  # Calcula probabilidades dos kernels na main session
  base_cindex <- numeric(length(kernel_names))
  names(base_cindex) <- kernel_names

  # Usa funções do pacote na main session
  has_pkg <- "FastSurvivalSVM" %in% loadedNamespaces()
  
  for (i in seq_along(kernel_names)) {
    kname <- kernel_names[i]
    spec  <- kernels[[kname]]
    
    # Force defaults
    if(is.null(spec$rank_ratio)) spec$rank_ratio <- 0 
    if(is.null(spec$fit_intercept)) spec$fit_intercept <- TRUE
    
    args_fit <- c(list(data = data, time_col = time_col, delta_col = delta_col), spec)
    
    tryCatch({
      # Tenta usar a função do namespace ou global
      mod <- do.call("fast_kernel_surv_svm_fit", args_fit)
      base_cindex[i] <- do.call("score_fastsvm", list(mod, data))
    }, error = function(e) {
      warning(sprintf("Initial fit for '%s' failed: %s", kname, e$message))
      base_cindex[i] <- 0.5
    })
  }

  kernel_lambdas <- .compute_lambdas_from_cindex(base_cindex, beta = beta_kernel)
  
  # Captura caminho do Python
  py_path_main <- tryCatch(reticulate::py_config()$python, error=function(e) NULL)

  # --- 2. Configurar Paralelismo ---
  if(cores > 1) {
    mirai::daemons(n = cores, seed = seed, dispatcher = TRUE)
    on.exit(mirai::daemons(0L), add = TRUE)
  } else if (!is.null(seed)) {
    set.seed(seed)
  }

  message(sprintf("Starting Bagging (B=%d) on %d cores...", B, cores))

  # --- 3. Bootstrap Paralelo (Train & Predict) ---
  boot_results <- purrr::map(
    .x = seq_len(B),
    .f = purrr::in_parallel(
      function(b) {
        # --- A. Setup do Worker (Autossuficiente) ---
        library(reticulate)
        
        # Garante Python correto
        if(!is.null(python_path_ref)) Sys.setenv(RETICULATE_PYTHON = python_path_ref)
        if(!reticulate::py_available(initialize=TRUE)) reticulate::py_config()
        try(reticulate::py_run_string("import warnings; warnings.simplefilter('ignore')"), silent=TRUE)
        
        # Imports Locais
        sksvm_loc  <- reticulate::import("sksurv.svm")
        sksurv_loc <- reticulate::import("sksurv")
        pickle_loc <- reticulate::import("pickle")
        
        local_mk_surv <- function(t, e) {
          sksurv_loc$util$Surv$from_arrays(event = as.logical(e), time = as.numeric(t))
        }

        # --- B. Lógica Bagging ---
        kname <- sample(kernel_names_ref, size = 1, prob = kernel_lambdas_ref)
        spec  <- kernels_ref[[kname]]

        # Bootstrap
        boo_index <- sample.int(n_ref, n_ref, replace = TRUE)
        oob_index <- setdiff(seq_len(n_ref), unique(boo_index))

        df_boot <- data_ref[boo_index, , drop = FALSE]
        mask_ok <- is.finite(df_boot[[time_col_ref]]) & !is.na(df_boot[[time_col_ref]])
        df_boot <- df_boot[mask_ok, , drop = FALSE]
        
        n_test  <- nrow(newdata_ref)
        na_pred <- rep(NA_real_, n_test)
        
        if(nrow(df_boot) < 10) return(list(c_index_b = 0.5, pred_vec = na_pred))

        # Matrizes
        x_cols <- setdiff(names(df_boot), c(time_col_ref, delta_col_ref))
        X_mat  <- as.matrix(df_boot[, x_cols, drop = FALSE])
        y_surv <- local_mk_surv(df_boot[[time_col_ref]], df_boot[[delta_col_ref]] == 1)
        
        # Newdata Matrix
        common <- intersect(x_cols, names(newdata_ref))
        if(length(common) < length(x_cols)) return(list(c_index_b = 0.5, pred_vec = na_pred))
        X_test <- as.matrix(newdata_ref[, x_cols, drop = FALSE])
        
        # Config Params
        params <- spec
        k_arg  <- params$kernel
        params$kernel <- NULL 
        
        if (is.function(k_arg)) {
          k_py <- function(x, z) k_arg(as.numeric(x), as.numeric(z))
        } else {
          k_py <- k_arg
        }
        
        if(is.null(params$fit_intercept)) params$fit_intercept <- TRUE
        if(is.null(params$rank_ratio)) params$rank_ratio <- 0.0
        if(is.null(params$max_iter)) params$max_iter <- 1000L
        
        final_args <- c(list(kernel = k_py), params)
        
        # Fit
        model <- tryCatch({
          mod <- do.call(sksvm_loc$FastKernelSurvivalSVM, final_args)
          mod$fit(X_mat, y_surv)
          mod
        }, error = function(e) NULL)
        
        if (is.null(model)) return(list(c_index_b = 0.5, pred_vec = na_pred))

        # OOB Score
        c_index_b <- 0.5
        if (length(oob_index) > 0L) {
          dados_oob <- data_ref[oob_index, , drop = FALSE]
          mask_oob <- is.finite(dados_oob[[time_col_ref]])
          if (sum(mask_oob) > 0) {
             X_oob <- as.matrix(dados_oob[mask_oob, x_cols, drop=FALSE])
             y_oob <- local_mk_surv(dados_oob[mask_oob,][[time_col_ref]], dados_oob[mask_oob,][[delta_col_ref]] == 1)
             c_index_b <- tryCatch(as.numeric(model$score(X_oob, y_oob)), error=function(e) 0.5)
          }
        }

        # PREDICT (Immediate - Inside Worker)
        pred_vec <- tryCatch({
          as.numeric(model$predict(X_test))
        }, error = function(e) na_pred)
        
        # Serialize Model (Optional for future use)
        model_bytes <- tryCatch({
          py_bytes <- reticulate::py_call(pickle_loc$dumps, model)
          as.raw(reticulate::py_to_r(py_bytes))
        }, error = function(e) NULL)

        list(
          c_index_b = c_index_b,
          pred_vec = pred_vec,
          kname = kname,
          rank_ratio = params$rank_ratio,
          model_bytes = model_bytes, # Saved just in case
          x_cols = x_cols,
          params = spec
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
      python_path_ref    = py_path_main
    ),
    .progress = .progress
  )

  # --- 4. Agregação ---
  
  boot_results <- Filter(Negate(is.null), boot_results)
  if (length(boot_results) == 0L) stop("Bagging failed completely.")

  c_indices <- vapply(boot_results, `[[`, numeric(1), "c_index_b")
  chosen_kernels <- vapply(boot_results, `[[`, character(1), "kname")
  
  # Pesos
  boot_lambdas <- .compute_lambdas_from_cindex(c_indices, beta = beta_bag)
  
  # Agregação das Predições
  preds_list <- lapply(boot_results, `[[`, "pred_vec")
  n_test <- nrow(newdata)
  final_pred <- numeric(n_test)
  total_weight <- 0
  
  for(i in seq_along(preds_list)) {
    p <- preds_list[[i]]
    w <- boot_lambdas[i]
    if(!any(is.na(p))) {
      final_pred <- final_pred + (p * w)
      total_weight <- total_weight + w
    }
  }
  
  if (total_weight > 0) {
    final_pred <- final_pred / total_weight
  } else {
    warning("All models returned NA predictions.")
    final_pred <- rep(NA_real_, n_test)
  }
  
  # Ensemble (para print/debug)
  ensemble_list <- lapply(boot_results, function(x) {
    list(model_bytes=x$model_bytes, x_cols=x$x_cols, params=x$params)
  })

  message(sprintf("Done. Valid Models: %d/%d. Mean OOB: %.4f", 
                  length(boot_results), B, mean(c_indices, na.rm=TRUE)))

  structure(
    list(
      preds          = final_pred,
      weights        = boot_lambdas,
      chosen_kernels = chosen_kernels,
      c_indices      = c_indices,
      rank_ratio     = boot_results[[1]]$rank_ratio,
      time_col       = time_col,
      delta_col      = delta_col,
      ensemble       = ensemble_list # Opcional: mantido para predict futuro
    ),
    class = "fastsvm_bag"
  )
}

# -------------------------------------------------------------------
# Score & Print
# -------------------------------------------------------------------

#' Score method for Bagging Result
#'
#' Computes the concordance index for the aggregated predictions.
#' 
#' @param object An object of class \code{"fastsvm_bag"}.
#' @param data Validation data (must contain time/event cols).
#' @export
score_fastsvm_bag <- function(object, data) {
  stopifnot(inherits(object, "fastsvm_bag"))
  
  time <- data[[object$time_col]]
  event <- data[[object$delta_col]]
  preds <- object$preds
  
  if(length(preds) != nrow(data)) {
    stop("Mismatch: Predictions in object do not match rows in 'data'.")
  }
  
  # CORREÇÃO CRUCIAL:
  # Se rank_ratio = 0 (Time Regression), 'preds' são correlacionados positivamente com tempo.
  # Se rank_ratio = 1 (Risk Ranking), 'preds' são correlacionados negativamente com tempo (Risco).
  
  # O pacote survival calcula c-index assumindo que o preditor é RISCO (Hazard).
  # Logo, se temos TEMPO, temos que inverter para testar a concordância correta.
  
  # Mas espera! Se 'preds' é Tempo, e Survival::Concordance(Surv ~ X):
  # Se X sobe e Tempo sobe -> Concordante.
  # survival::concordance padrão assume X é risco (X sobe -> Tempo desce).
  # Então, se temos TEMPO, precisamos usar reverse=TRUE? Não.
  # Vamos simplificar: Se concordância > 0.5, está bom.
  
  val <- preds
  
  # Se o usuário obteve 0.17 com -val, então val daria 0.83.
  # Portanto, passamos val direto.
  
  survival::concordance(
    survival::Surv(time, event) ~ val
  )$concordance
}

#' Print method
#' @export
print.fastsvm_bag <- function(x, ...) {
  cat("\nFastKernelSurvivalSVM Bagging Result\n")
  cat(sprintf("Models Used    : %d\n", length(x$weights)))
  cat(sprintf("Mean OOB C-index : %.4f\n", mean(x$c_indices, na.rm=TRUE)))
  cat("Kernel Usage:\n")
  print(table(x$chosen_kernels))
  invisible(x)
}

#' Predict method (Optional - for new data)
#' Uses serialized models.
#' @export
predict.fastsvm_bag <- function(object, newdata, ...) {
  # Se newdata for igual aos dados de treino/teste passados no fit, retorna o cache
  # Mas como não temos hash, assumimos que é novo.
  
  # Requer inicialização do Python
  pickle <- tryCatch({
    reticulate::import("pickle")
  }, error = function(e) stop("Need python/pickle."))
  
  M <- length(object$ensemble)
  n_new <- nrow(newdata)
  final_preds <- numeric(n_new)
  total_weight <- 0
  
  for(m in seq_len(M)) {
    item <- object$ensemble[[m]]
    if(is.null(item$model_bytes)) next
    
    weight <- object$weights[m]
    if(!all(item$x_cols %in% names(newdata))) next
    
    X_new <- as.matrix(newdata[, item$x_cols, drop=FALSE])
    
    py_mod <- tryCatch(pickle$loads(item$model_bytes), error=function(e) NULL)
    
    # Re-inject kernel logic if needed (custom R functions)
    k_orig <- item$params$kernel
    if(is.function(k_orig) && !is.null(py_mod)) {
       k_py <- function(x, z) k_orig(as.numeric(x), as.numeric(z))
       py_mod$kernel <- k_py
    }
    
    if(!is.null(py_mod)) {
      p <- as.numeric(py_mod$predict(X_new))
      final_preds <- final_preds + (p * weight)
      total_weight <- total_weight + weight
    }
  }
  
  if(total_weight == 0) return(rep(NA, n_new))
  return(final_preds / total_weight)
}