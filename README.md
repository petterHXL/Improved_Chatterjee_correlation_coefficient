# Chatterjee's Correlation Coefficient Project

A comprehensive implementation and analysis of Chatterjee's correlation coefficient (CCC), including comparisons with Pearson's and Spearman's correlations across various dependence structures.

## Overview

This project implements Chatterjee's correlation coefficient and conducts extensive power comparisons with traditional correlation measures (Pearson's and Spearman's) under different dependence scenarios, noise levels, and sample sizes.

## Features

- **Original Chatterjee's Correlation**: Implementation of the standard CCC algorithm
- **M-NN Extension**: Multiple nearest neighbors version for improved power
- **Normalized CCC**: Normalized version with proper bounds
- **Normalized M-NN CCC**: Normalized M-NN Chatterjee's correlation coefficient for comparability across datasets
- **Inverse Distance Weighted CCC**: Global inverse distance weighted version for improved detection of complex dependencies
- **Normalized IDW CCC**: Normalized inverse distance weighted version for fair comparison across datasets
- **Power Comparison Framework**: Comprehensive simulation framework
- **Multiple Dependence Scenarios**: Linear, quadratic, sinusoidal, piecewise (step), mixed, and heteroscedastic relationships

### Toolbox Functions (in `code/toolbox/chatterjee_correlation.py`)
- **chatterjee_cc(x, y):** Computes the original Chatterjee's correlation coefficient between two variables.
- **normalized_chatterjee_cc(x, y):** Computes a normalized version of CCC, scaling the coefficient for better interpretability.
- **chatterjee_cc_mnn_with_ties(x, y, M):** Computes the M-NN extension of CCC, which uses multiple right nearest neighbors to improve statistical power and efficiency, with robust handling of ties. M is a user-chosen positive integer (e.g., 3, 5, 10, or sqrt(n)).
- **normalized_chatterjee_cc_mnn(x, y, M):** Computes the normalized M-NN Chatterjee's correlation coefficient. Normalization is performed as xi'_M = max(-1, xi_M(x, y) / xi_M(y, y)), where xi_M(y, y) is the maximum possible value (perfect dependence). This ensures the normalized value is in [-1, 1] and comparable across datasets.
- **inverse_distance_weighted_chatterjee(x, y):** Computes the inverse distance weighted Chatterjee's correlation coefficient. This improved version uses global inverse distance weighted rank differences: T_n = sum_{i=1}^{n-1} sum_{j=i+1}^{n} |R_j - R_i| / (j-i), normalized by H_n = (n+1)/3 * sum_{i != j} 1/|i-j|. Particularly effective for detecting oscillating and complex non-monotonic dependencies.
- **normalized_inverse_distance_weighted_chatterjee(x, y):** Computes the normalized inverse distance weighted Chatterjee's correlation coefficient. Normalization is performed as xi'_IM = max(-1, xi_IM(x, y) / xi_IM(y, y)), ensuring values are in [-1, 1] and comparable across datasets.

All functions require only `numpy` and `scipy` as dependencies.

## Project Structure

```
Improved_Chatterjee_correlation_coefficient/
├── code/
│   ├── experiment code/
│   │   ├── linear_with_noise.py
│   │   ├── linear_without_noise.py
│   │   ├── quadratic_with_noise.py
│   │   ├── quadratic_without_noise.py
│   │   ├── sinusoidal_with_noise.py
│   │   ├── sinusoidal_without_noise.py
│   │   ├── piecewise_with_noise.py
│   │   ├── piecewise_without_noise.py
│   │   ├── mixed_with_noise.py
│   │   ├── mixed_without_noise.py
│   │   ├── heteroscedastic.py
│   │   └── power_comparison.py
│   └── toolbox/
│       └── chatterjee_correlation.py
├── Results/
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/petterHXL/Improved_Chatterjee_correlation_coefficient.git
cd Improved_Chatterjee_correlation_coefficient
```

2. Install required dependencies:
```bash
pip3 install numpy scipy matplotlib seaborn pandas tqdm xicorpy
```

## Usage

### Basic Usage

```python
from code.toolbox.chatterjee_correlation import chatterjee_cc, normalized_chatterjee_cc

# Compute Chatterjee's correlation
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]  # Quadratic relationship
ccc = chatterjee_cc(x, y)
norm_ccc = normalized_chatterjee_cc(x, y)

print(f"Chatterjee's CC: {ccc:.3f}")
print(f"Normalized CC: {norm_ccc:.3f}")
```

### Running Experiments

#### Individual Relationship Tests
```bash
cd "code/experiment code"
python3 linear_without_noise.py
python3 linear_with_noise.py
python3 quadratic_without_noise.py
python3 quadratic_with_noise.py
python3 sinusoidal_without_noise.py
python3 sinusoidal_with_noise.py
python3 piecewise_without_noise.py
python3 piecewise_with_noise.py
python3 mixed_without_noise.py
python3 mixed_with_noise.py
python3 heteroscedastic.py
```

#### Power Comparison Experiment
```bash
cd "code/experiment code"
python3 power_comparison.py
```

This will run a comprehensive power comparison across:
- **Scenarios**: Linear, piecewise (step), quadratic, sinusoidal, heteroscedastic, mixed
- **Sample sizes**: 20, 100, 500
- **Noise levels**: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
- **Methods**: Chatterjee's, Pearson's, Spearman's correlations


## M-NN Extension

The project also includes an implementation of the M-NN (Multiple Nearest Neighbors) extension of Chatterjee's correlation:

```python
from code.toolbox.chatterjee_correlation import chatterjee_cc_mnn_with_ties, normalized_chatterjee_cc_mnn

# M-NN version with M = sqrt(n)
M = int(np.sqrt(len(x)))
ccc_mnn = chatterjee_cc_mnn_with_ties(x, y, M)
norm_ccc_mnn = normalized_chatterjee_cc_mnn(x, y, M)
print(f"M-NN Chatterjee's CC (M={M}): {ccc_mnn:.3f}")
print(f"Normalized M-NN Chatterjee's CC (M={M}): {norm_ccc_mnn:.3f}")
```

- **How to choose M:** M is a user-chosen positive integer. M=1 recovers the original CCC; larger M increases the locality considered. Typical values are 3, 5, 10, or sqrt(n).
- **Normalization:** The normalized M-NN CCC divides the observed value by its maximum possible value (when x = y), ensuring comparability across datasets and settings:

  xi'_M = max(-1, xi_M(x, y) / xi_M(y, y))

This extension provides improved power for detecting subtle dependencies and a normalized version for fair comparison.

## Output

The power comparison experiment generates:
- **CSV file**: `power_comparison_results.csv` with all simulation results
- **Plots**: Power vs. noise level and sample size effects for each scenario
- **Summary tables**: Easy-to-read comparisons across methods

## Contributing

Feel free to contribute by:
- Adding new dependence scenarios
- Implementing additional correlation measures
- Improving the visualization and analysis
- Adding unit tests

## License

This project is open source and available under the MIT License.

## References

Please cite the following papers if you use this toolbox in your research:

Chatterjee, S. (2021). A new coefficient of correlation. *Journal of the American Statistical Association*, 116(536), 2009-2022. https://doi.org/10.1080/01621459.2020.1758115

Dalitz, C., Arning, J., & Goebbels, S. (2024). A Simple Bias Reduction for Chatterjee's Correlation. *Journal of Statistical Theory and Practice*, 18(4), Article 51. https://doi.org/10.1007/s42519-024-00399-y

Lin, Z., & Han, F. (2023). On boosting the power of Chatterjee's rank correlation. *Biometrika*, 110(2), 283–299. https://doi.org/10.1093/biomet/asac048

Xia, L., Cao, R., Du, J., & Chen, X. (2025). The improved correlation coefficient of Chatterjee. *Journal of Nonparametric Statistics*, 37(2), 265–281. https://doi.org/10.1080/10485252.2024.2373242

---

For questions or contributions, please open an issue or pull request on [GitHub](https://github.com/petterHXL/Improved_Chatterjee_correlation_coefficient).
