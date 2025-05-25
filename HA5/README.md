# PAC-Bayes Experiments: Figure 2a Reproduction

This script reproduces the data behind Figure 2a of Thiemann, Igel, Wintenberger & Seldin "A Strongly Quasiconvex PAC-Bayesian Bound" (arXiv:1608.05610v2).

## Overview

The script follows exactly the construction in Sections 5 and 6 of the paper:

- **Build an ensemble H of m weak RBF-SVMs**
  - Each is trained on r = d+1 examples
  - Validated on the other n-r examples
- **Compute the validation-loss vector L̂_val(h,S)** (Eq. 13)
- **Minimize the PAC-Bayes-lambda bound** of Theorem 6 by the alternating (rho,lambda) procedure derived from Eqs. (7) and (8)
  - Numerically-stable computation in log-domain
- **Evaluate on a disjoint test set** and draw the figure that matches Figure 2(a)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

The requirements include:

- numpy
- pandas
- matplotlib
- scipy
- scikit-learn

## Usage

### Basic Usage (Single Run)

```bash
python exercise3_8_script.py
```

### Multiple Runs for Confidence Intervals

```bash
python exercise3_8_script.py --num_runs 10
```

### Command Line Options

- `--num_runs`: Number of runs for confidence interval computation (default: 1)

## Examples

1. **Single run** (fast, no confidence intervals):

   ```bash
   python exercise3_8_script.py --num_runs 1
   ```

2. **10 runs** (moderate time, with 95% confidence intervals):

   ```bash
   python exercise3_8_script.py --num_runs 10
   ```

3. **50 runs** (slower, but more robust confidence intervals):

   ```bash
   python exercise3_8_script.py --num_runs 50
   ```

## Output

The script generates:

- **Figure**: `figure_2a_runs_{num_runs}.pdf` - Reproduction of Figure 2(a)
- **Console output**: Statistics in LaTeX format including:
  - Method performance vs CV SVM baseline
  - PAC-Bayes bound values
  - Runtime comparisons
  - Confidence intervals (when num_runs > 1)
