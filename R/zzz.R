# R/zzz.R

sksurv <- NULL
sksvm  <- NULL
np     <- NULL

.onLoad <- function(libname, pkgname) {
  # Declarar as dependências Python do pacote
  reticulate::py_require(
    packages = c("numpy", "scikit-learn", "scikit-survival", "pandas")
  )
  
  # Importar módulos com delay_load = TRUE
  # (Python só é inicializado quando o usuário realmente usar)
  np     <<- reticulate::import("numpy",       delay_load = TRUE)
  sksurv <<- reticulate::import("sksurv",      delay_load = TRUE)
  sksvm  <<- reticulate::import("sksurv.svm",  delay_load = TRUE)
}

#' @importFrom utils globalVariables
# Silence R CMD check notes regarding variables in mirai/purrr context
if (getRversion() >= "2.15.1") {
  utils::globalVariables(c(
    "python_path_ref",
    "kernel_names_ref",
    "kernel_lambdas_ref",
    "kernels_ref",
    "n_ref",
    "data_ref",
    "time_col_ref",
    "newdata_ref",
    "delta_col_ref",
    "mtry_ref"
  ))
}
