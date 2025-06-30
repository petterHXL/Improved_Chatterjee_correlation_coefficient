# Chatterjee's Correlation Coefficient Project

A comprehensive implementation and analysis of Chatterjee's correlation coefficient (CCC), including comparisons with Pearson's and Spearman's correlations across various dependence structures.

## Overview

This project implements Chatterjee's correlation coefficient and conducts extensive power comparisons with traditional correlation measures (Pearson's and Spearman's) under different dependence scenarios, noise levels, and sample sizes.

## Features

- **Original Chatterjee's Correlation**: Implementation of the standard CCC algorithm
- **M-NN Extension**: Multiple nearest neighbors version for improved power
- **Normalized CCC**: Normalized version with proper bounds
- **Power Comparison Framework**: Comprehensive simulation framework
- **Multiple Dependence Scenarios**: Linear, quadratic, sinusoidal, piecewise (step), mixed, and heteroscedastic relationships

### Toolbox Functions (in `code/toolbox/chatterjee_correlation.py`)
- **chatterjee_cc(x, y):** Computes the original Chatterjee's correlation coefficient between two variables.
- **normalized_chatterjee_cc(x, y):** Computes a normalized version of CCC, scaling the coefficient for better interpretability.
- **chatterjee_cc_mnn_with_ties(x, y, M):** Computes the M-NN extension of CCC, which uses multiple right nearest neighbors to improve statistical power and efficiency, with robust handling of ties.
- **find_m_right_neighbors_robust(x, y, i, M):** Helper function for robust neighbor finding with tie-breaking.

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

## Dependence Scenarios

### 1. Linear Relationship
- **Script**: `linear_without_noise.py`, `linear_with_noise.py`
- **Formula**: `Y = 2X + 1 (+ noise)`
- **Expected**: Pearson's and Spearman's should perform best

### 2. Piecewise (Step Function) Relationship
- **Script**: `piecewise_without_noise.py`, `piecewise_with_noise.py`
- **Formula**: Piecewise monotonic function with jumps (see code for details)
- **Expected**: Spearman's should perform best (monotonic)

### 3. Quadratic Relationship
- **Script**: `quadratic_without_noise.py`, `quadratic_with_noise.py`
- **Formula**: `Y = 3(X² - 0.5) + 2λ·ε`
- **Expected**: Chatterjee's should perform best (non-monotonic)

### 4. Sinusoidal Relationship
- **Script**: `sinusoidal_without_noise.py`, `sinusoidal_with_noise.py`
- **Formula**: `Y = cos(6πX) + 3λ·ε`
- **Expected**: Chatterjee's should perform best (oscillatory)

### 5. Heteroscedastic Relationship
- **Script**: `heteroscedastic.py`
- **Formula**: `Y = X * noise` (variance depends on X)
- **Expected**: Chatterjee's should perform best (variance dependence)

### 6. Mixed Relationship
- **Script**: `mixed_without_noise.py`, `mixed_with_noise.py`
- **Formula**: Combination of different functional forms (see code for details)
- **Expected**: Chatterjee's is robust to complex dependencies

## Key Findings

Based on the power comparison experiments:

- **Chatterjee's CCC** performs best for non-monotonic relationships (quadratic, sinusoidal, mixed, heteroscedastic)
- **Pearson's PCC** performs best for linear relationships
- **Spearman's SCC** performs best for monotonic relationships (piecewise)
- Power decreases with increasing noise for all methods
- Power increases with sample size for all methods

## M-NN Extension

The project also includes an implementation of the M-NN (Multiple Nearest Neighbors) extension of Chatterjee's correlation:

```python
from code.toolbox.chatterjee_correlation import chatterjee_cc_mnn_with_ties

# M-NN version with M = sqrt(n)
M = int(np.sqrt(len(x)))
ccc_mnn = chatterjee_cc_mnn_with_ties(x, y, M)
```

This extension provides improved power for detecting subtle dependencies.

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

Chatterjee, S. (2021). A new coefficient of correlation. *Journal of the American Statistical Association*, 116(536), 2009-2022. https://doi.org/10.1080/01621459.2021.1932098

Seshadri, V., & Chatterjee, S. (2022). A simple bias reduction for Chatterjee's correlation. *Journal of Multivariate Analysis*, 189, 104880. https://doi.org/10.1016/j.jmva.2022.104880

Zhu, X., & Li, J. (2024). On boosting the power of Chatterjee's rank correlation. *Journal of the American Statistical Association*, 119(545), 1-14. https://doi.org/10.1080/01621459.2023.2253692

---

For questions or contributions, please open an issue or pull request on [GitHub](https://github.com/petterHXL/Improved_Chatterjee_correlation_coefficient).
