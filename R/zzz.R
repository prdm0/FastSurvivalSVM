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
