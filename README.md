# FastSurvivalSVM <img src="logo.png" align="right" width="250"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)

> **High-Performance Kernel Survival Support Vector Machines for R**

**FastSurvivalSVM** bridges the gap between R and the state-of-the-art **FastKernelSurvivalSVM** implementation from the Python library `scikit-survival`.

It goes beyond standard modeling by introducing **Random Machines**, a powerful ensemble method that combines bagging, random feature subspaces, and adaptive kernel selection to maximize predictive performance on right-censored data.

---

## ⚡ Why FastSurvivalSVM?

- 🚀 **Speed & Efficiency:** Hybrid parallel execution using `mirai` (for R kernels) and optimized C++/Python threads (for native kernels).
- 🧠 **Random Machines:** A novel ensemble algorithm that automatically selects the best kernel for each bootstrap sample using an internal holdout strategy.
- 🎨 **Custom Kernels Made Easy:** Define custom kernels using simple R functions via the `grid_kernel()` helper—no complex closures required.
- 🎛️ **Unified Tuning:** Tune native and custom kernels simultaneously with `tune_random_machines()`.
- 🛠️ **Zero Configuration:** Automatic Python environment handling via `reticulate`.

---

## 📦 Installation

Install the development version from GitHub:

```r
# install.packages("remotes")
remotes::install_github("prdm0/FastSurvivalSVM", force = TRUE)
```

Note: On the first run, the package will automatically set up a minimal Python environment. No manual Python installation is required.

## 🏁 Quick Start: Single Model

Fit a standard Survival SVM using a built-in kernel.

```r
library(FastSurvivalSVM)

# 1. Generate synthetic survival data
set.seed(42)
df <- data_generation(n = 200, prop_cen = 0.25)

# 2. Fit the model (Regression mode)
fit <- fastsvm(
  data      = df,
  time_col  = "tempo",
  delta_col = "cens",
  kernel    = "rbf",   # Native Scikit-learn kernel
  alpha     = 1,
  rank_ratio = 0       # 0 = Regression, 1 = Ranking
)

# 3. Predict & Score
pred <- predict(fit, df)
c_index <- score(fit, df)

cli::cli_alert_success("C-index: {round(c_index, 4)}")
```

## 🎨 Custom Kernels: The grid_kernel Way

Forget complex function factories. With FastSurvivalSVM, you define the math, and grid_kernel() handles the rest.

### 1. Define the Math

```r
# Example: A Wavelet Kernel
my_wavelet <- function(x, z, A = 1) {
  u <- (as.numeric(x) - as.numeric(z)) / A
  prod(cos(1.75 * u) * exp(-0.5 * u^2))
}
```

### 2. Instantiate and Fit

```r
# Create a specific instance with A = 0.5
wav_instance <- grid_kernel(my_wavelet, A = 0.5)

fit_custom <- fastsvm(
  data      = df,
  time_col  = "tempo",
  delta_col = "cens",
  kernel    = wav_instance,
  alpha     = 1
)
```

## 🎛️ Hyperparameter Tuning

The package offers a robust tuning engine that supports Hybrid Parallelism: it uses Python threads for native kernels and R background processes for custom kernels simultaneously.

```r
# Define the Kernel Mix (Structure)
kernel_mix <- list(
  rbf_std    = list(kernel = "rbf", rank_ratio = 0),
  wavelet_ok = list(rank_ratio = 0)
)

# Define Parameter Grids
param_grids <- list(
  # Native: Tune gamma and alpha
  rbf_std = list(
    gamma = c(0.01, 0.1),
    alpha = c(0.1, 1)
  ),

  # Custom: Tune 'A' (via grid_kernel) and alpha
  wavelet_ok = list(
    kernel = grid_kernel(my_wavelet, A = c(0.5, 1.0, 2.0)),
    alpha  = c(0.1, 1)
  )
)

# Run Hybrid Tuning
tune_res <- tune_random_machines(
  data        = df,
  time_col    = "tempo",
  delta_col   = "cens",
  kernel_mix  = kernel_mix,
  param_grids = param_grids,
  cv          = 3,
  cores       = parallel::detectCores()
)
```

## 🔥 Random Machines

Random Machines allows you to leverage the power of multiple kernels in a single strong ensemble. It uses Bagging + Random Subspace (mtry) + Adaptive Kernel Selection.

```r
# 1. Prepare optimized kernels from tuning result
final_kernels <- as_kernels(tune_res, kernel_mix)

# 2. Train the Ensemble
ens_model <- random_machines(
  data         = df,
  newdata      = df, # Usually a test set
  time_col     = "tempo",
  delta_col    = "cens",
  kernels      = final_kernels,
  B            = 50,   # Number of machines
  crop         = 0.10, # Pruning threshold (drop weak kernels)
  mtry         = NULL, # Random feature subspace
  cores        = parallel::detectCores()
)

# 3. Inspect Results
print(ens_model)
```

## 📚 References

Pölsterl, S. et al. (2016).

An Efficient Training Algorithm for Kernel Survival Support Vector Machines. arXiv:1611.07054.

## 📄 License

MIT © Pedro Rafael Diniz Marinho & Agatha Sacramento Rodrigues.
