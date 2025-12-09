# FastSurvivalSVM <img src="logo.png" align="right" width="250"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)

> **High-Performance Kernel Survival Support Vector Machines for R**

**FastSurvivalSVM** provides a seamless R interface to the state-of-the-art **FastKernelSurvivalSVM** implementation from the Python library `scikit-survival`.

Beyond standard modeling, it introduces **Random Machines**, a powerful ensemble method that combines bagging, random feature subspaces, and adaptive kernel selection to maximize predictive performance on right-censored data.

---

## ⚡ Why FastSurvivalSVM?

- 🚀 **Speed & Efficiency:** Parallel execution powered by `mirai` and optimized Python backend.
- 🧠 **Random Machines:** Ensemble algorithm that automatically selects the best kernels using internal holdout.
- ✂️ **Adaptive Pruning:** The `crop` parameter discards weak kernels from the ensemble.
- 🎨 **Custom Kernels in R:** Define fully custom kernel functions (e.g., Wavelet) using closures and function factories.
- 🛠️ **Zero Configuration:** Automatic Python environment handling via `reticulate`.
- ✨ **Modern Output:** Clean, readable console summaries with `cli`.

---

## 📦 Installation

Install the development version from GitHub:

```r
# install.packages("remotes")
remotes::install_github("prdm0/FastSurvivalSVM", force = TRUE)
```

> **Note:** On the first run, the package will automatically set up a minimal Python environment.  
No manual Python installation required.

---

## 🏁 Quick Start: Single Model

Fit a standard Survival SVM using a built-in kernel.

```r
library(FastSurvivalSVM)

# 1. Generate synthetic survival data
set.seed(123)
df <- data_generation(n = 100, prop_cen = 0.3)

# 2. Fit the model
fit <- fastsvm(
  data      = df,
  time_col  = "tempo",
  delta_col = "cens",
  kernel    = "rbf",
  alpha     = 1
)

# 3. Predict & score
pred <- predict(fit, df)
c_index <- score(fit, df)

print(c_index)
#> [1] 0.785
```

---

## 🔥 Random Machines

Random Machines is an ensemble technique that:

- reduces variance,
- optimizes kernel choice,
- assigns kernel weights through an internal holdout strategy,
- and aggregates predictions using weighted voting.

Use:

- `crop` to eliminate weak kernels,
- `mtry` to select random feature subsets, similar to Random Forest.

### Example

```r
# Candidate kernels (mix strings and custom R functions)
kernels_list <- list(
  linear = list(kernel = "linear", alpha = 1),
  rbf    = list(kernel = "rbf", alpha = 0.5, gamma = 0.1),
  poly   = list(kernel = "poly", degree = 2, alpha = 1)
)

# Run Random Machines
ens_model <- random_machines(
  data         = df,
  newdata      = df,   # usually your test set
  time_col     = "tempo",
  delta_col    = "cens",
  kernels      = kernels_list,
  B            = 50,
  crop         = 0.15,
  mtry         = NULL,
  cores        = 4
)

# Inspect results
print(ens_model)
```

---

## 🎨 Advanced: Custom Kernels in R

Fully customize kernel functions via function factories (closures), ideal for kernels with hyperparameters.

### Example: Wavelet Kernel

```r
make_wavelet <- function(A = 1) {
  force(A)
  function(x, z) {
    x <- as.numeric(x)
    z <- as.numeric(z)
    u <- (x - z) / A
    prod(cos(1.75 * u) * exp(-0.5 * u^2))
  }
}

# Use custom kernel
fit_wav <- fastsvm(
  data      = df,
  time_col  = "tempo",
  delta_col = "cens",
  kernel    = make_wavelet(A = 1.5)
)
```

---

## 📚 References

- **Pölsterl, S. et al. (2016)**.  
  *An Efficient Training Algorithm for Kernel Survival Support Vector Machines.*  
  arXiv:1611.07054.

---

## 📄 License

MIT © Pedro Rafael Diniz Marinho & Agatha Sacramento Rodrigues.
