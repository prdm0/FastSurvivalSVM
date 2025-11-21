# remotes::install_github("prdm0/FastSurvivalSVM", force = T)

library(FastSurvivalSVM)

data_generation <- function(n, prop_cen) {
    # 3 covariaveis
    x1 <- rnorm(n, 1, 1)
    x2 <- rnorm(n, 2, 2)
    x3 <- rexp(n)
    # parâmetros da Weibull
    xbeta <- x1 * log(abs(x2)) + 2 * sin(x3 - x2)^2
    shape <- 2
    scale <- 5
    # tempos verdadeiros (sem censura)
    u <- runif(n, 0, 1)
    time_t <- ((-log(1 - u))^(1 / shape)) * scale * exp(-xbeta)
    # selecionar quem será censurado e gerar os tempos
    ind_cens <- sample(1:n, n * prop_cen, replace = FALSE)
    time_gerado <- time_t
    time_gerado[ind_cens] <- runif(
        length(ind_cens),
        min(time_t),
        time_t[ind_cens]
    )
    # criando a indicadora de falha
    cens <- rep(1, n)
    cens[ind_cens] <- 0

    dados_g <- data.frame(
        tempo = time_gerado,
        cens = cens,
        x1 = x1,
        x2 = x2,
        x3 = x3
    )
    return(dados_g)
}

set.seed(123)

df <- data_generation(n = 300L, prop_cen = 0.1)

## -----------------------------------------
## Exemplo de uso do kernel wavelet com FastSurvivalSVM
## -----------------------------------------

library(FastSurvivalSVM)

## 1) Dados de exemplo
set.seed(123)
n <- 150
df <- data.frame(
  time   = rexp(n, rate = 0.1),
  status = rbinom(n, 1, 0.7),  # 1 = evento, 0 = censura
  x1     = rnorm(n),
  x2     = rnorm(n),
  x3     = rnorm(n)
)

## 2) Função mãe wavelet (do artigo): h(u) = cos(1.75 u) * exp(-0.5 u^2)
wavelet_mother <- function(u) {
  cos(1.75 * u) * exp(-0.5 * u^2)
}

## 3) Kernel wavelet multidimensional:
##    K(x, z) = prod_k h( (x_k - z_k) / A )
wavelet_kernel <- function(x, z, A = 1) {
  x <- as.numeric(x)
  z <- as.numeric(z)
  stopifnot(length(x) == length(z))
  stopifnot(length(A) == 1L, A > 0)

  u <- (x - z) / A
  prod(wavelet_mother(u))
}

## (Opcional) "factory" para fixar A e usar direto no fast_kernel_surv_svm_fit
make_wavelet_kernel <- function(A = 1) {
  force(A)
  function(x, z) wavelet_kernel(x, z, A = A)
}

## 4) Ajuste do modelo FastKernelSurvivalSVM com kernel wavelet
fit_wavelet <- fast_kernel_surv_svm_fit(
  data       = df,
  time_col   = "time",
  delta_col  = "status",
  kernel     = make_wavelet_kernel(A = 0.5),  # A é um hiperparâmetro do kernel
  alpha      = 1,      # hiperparâmetro de regularização do SVM
  rank_ratio = 0,      # 0 = regressão (sobre tempo transformado)
  fit_intercept = FALSE
)

## 5) Uso do modelo

# Previsões (scores de risco ou tempos transformados, dependendo de rank_ratio)
preds <- predict(fit_wavelet, df)
head(preds)

# C-index nos próprios dados
cindex_wavelet <- score_fastsvm(fit_wavelet, df)
cindex_wavelet
