# FastSurvivalSVM

FastSurvivalSVM is an R package that provides a high-level interface to the Python implementation of **FastKernelSurvivalSVM**, available in the `scikit-survival` library. It enables R users to fit kernel-based survival Support Vector Machines for right-censored time-to-event data using a unified, user-friendly API.

This package uses the **reticulate** framework to communicate with Python, automatically handles the necessary Python dependencies, and makes the powerful kernel survival SVM models from `scikit-survival` accessible directly from R workflows.

---

## ✨ Features

- Fit **kernel survival SVMs** using the Python function  
  [`sksurv.svm.FastKernelSurvivalSVM`](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.svm.FastKernelSurvivalSVM.html)
- Handles **right-censored survival data**
- Supports:
  - Built-in scikit-learn kernels (`"rbf"`, `"poly"`, `"sigmoid"`, etc.)
  - **Custom kernels written in R**
- Returns predictions, concordance index (C-index), and detailed summaries
- Automatically constructs the Python `Surv` structured array required by scikit-survival
- Minimal Python setup required; dependencies are managed automatically with `reticulate::py_require()`

---

## 🔗 Repositories

- **R package:** https://github.com/prdm0/FastSurvivalSVM  
- **Python backend (`scikit-survival`):** https://scikit-survival.readthedocs.io/en/stable/index.html  
- **Function wrapped by this package:**  
  https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.svm.FastKernelSurvivalSVM.html  

---

## 📦 Installation

### 1. Install the R package (development version)

```r
remotes::install_github("prdm0/FastSurvivalSVM")
```

### 2. Python dependencies

You do **not** need to install Python manually in most cases.

The package automatically declares the following Python requirements:

- `numpy`
- `pandas`
- `scikit-learn`
- `scikit-survival`

These will be installed in an **ephemeral virtual environment** the first time the package is used.

If you prefer manual installation, configure Python before loading the package:

```r
library(reticulate)
use_python("/usr/bin/python3", required = TRUE)
```

---

## 🚀 Example: Fitting a Survival SVM

```r
library(FastSurvivalSVM)

set.seed(1)
n <- 100
df <- data.frame(
  time   = rexp(n, rate = 0.1),
  status = rbinom(n, 1, 0.7),  # 1 = event, 0 = censoring
  x1     = rnorm(n),
  x2     = rnorm(n)
)

# Fit with RBF kernel (default)
fit_rbf <- fast_kernel_surv_svm_fit(
  data       = df,
  time_col   = "time",
  delta_col  = "status",
  kernel     = "rbf",
  alpha      = 1,
  rank_ratio = 0  # regression mode
)

# Predictions
preds <- predict(fit_rbf, df)
head(preds)

# Concordance index
score_fastsvm(fit_rbf, df)

# Summary
summary(fit_rbf)
```

---

## 🎨 Example: Using a Custom Kernel in R

```r
# Custom RBF kernel
rbf_custom <- function(x, z, sigma = 0.5) {
  d2 <- sum((x - z)^2)
  exp(-d2 / (2 * sigma^2))
}

fit_custom <- fast_kernel_surv_svm_fit(
  data       = df,
  time_col   = "time",
  delta_col  = "status",
  kernel     = function(x, z) rbf_custom(x, z),
  alpha      = 1,
  rank_ratio = 0
)

summary(fit_custom)
```

---

## 📘 Model Details

The underlying Python estimator implements the kernel survival SVM described in:

> **Pölsterl, S., Navab, N., Katouzian, A. (2016).**  
> *An Efficient Training Algorithm for Kernel Survival Support Vector Machines*.  
> 4th Workshop on Machine Learning in Life Sciences.  
> arXiv:1611.07054

### Supported Kernel Types

You may pass:

| Type | Example | Description |
|------|---------|-------------|
| String | `"rbf"` | scikit-learn kernel |
| R function | `function(x,z) ...` | Must return scalar kernel value |
| `"precomputed"` | custom matrix | Kernel matrix supplied as `X` |

### Prediction Types

- If `rank_ratio = 1`: **risk scores** (higher = higher risk)
- If `rank_ratio < 1`: **transformed survival times** (higher = longer survival)

---

## 📊 Methods Provided

### `fast_kernel_surv_svm_fit()`
Fit the model.

### `predict.fastsvm()`
Predict risk scores or transformed survival times.

### `score_fastsvm()`
Compute C-index.

### `summary.fastsvm()`
Detailed model summary including:
- Coefficients
- Number of support vectors
- Kernel type
- Hyperparameters
- Optimization iterations

### `print.fastsvm()`
Compact printing.

---

## ⚙️ Internal Mechanics

### Python Bridge

The package internally creates:

- A Python callable kernel (bridging R → Python)
- A `Surv` structured array using:

```python
sksurv.util.Surv.from_arrays(event, time)
```

### Why only right censoring?

Because FastKernelSurvivalSVM is explicitly designed for:

- **Right-censored data**
- No interval or left censoring
- No competing risks

Matching scikit-survival’s theoretical formulation.

---

## 👨‍💻 Authors

- **Pedro Rafael Diniz Marinho**  
  *Author & Maintainer*  
  Email: pedro.rafael.marinho@gmail.com  

- **Agatha Sacramento Rodrigues**  
  *Author*  
  Email: agatha.srodrigues@gmail.com  

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## 🙌 Acknowledgements

This package is a high-level R wrapper inspired by the work of:

- Sebastian Pölsterl and contributors to **scikit-survival**
- The **reticulate** project (RStudio / Posit)
- The broader survival analysis community

---

If you want, I can also generate:

- **Hex sticker**  
- **CRAN-style pkgdown website**  
- **Vignette demonstrating real-world usage**  
- **Simulation study comparing kernels**

