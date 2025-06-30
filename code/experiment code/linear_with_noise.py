import numpy as np
import xicorpy as xicor
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from Improved_Chatterjee_correlation_coefficient.code.toolbox.chatterjee_correlation import chatterjee_cc, normalized_chatterjee_cc, chatterjee_cc_mnn_with_ties

# Linear relationship with noise
x = np.linspace(-5, 5, 100)
y_clean = 2 * x + 1  # Linear function: y = 2x + 1
noise_level = 1.0  # Adjust this to control noise amount
y = y_clean + np.random.normal(0, noise_level, len(x))

# Compute all correlations
xi_result = xicor.compute_xi_correlation(x, y, get_p_values=False)
if isinstance(xi_result, tuple):
    xi = xi_result[0]
else:
    xi = xi_result
cc = chatterjee_cc(x, y)
norm_cc = normalized_chatterjee_cc(x, y)
M = int(np.sqrt(len(x)))
ccc_mnn = chatterjee_cc_mnn_with_ties(x, y, M)

pearson_corr, _ = pearsonr(x, y)
spearman_corr, _ = spearmanr(x, y)

if hasattr(xi, 'item'):
    xi_scalar = xi.item()
else:
    xi_scalar = float(xi)

print(f"Xi correlation: {xi_scalar:.3f}")
print(f"Chatterjee's CC: {cc:.3f}")
print(f"Normalized Chatterjee's CC: {norm_cc:.3f}")
print(f"M-NN Chatterjee's CC (M={M}): {ccc_mnn:.3f}")
print(f"Pearson's CC: {pearson_corr:.3f}")
print(f"Spearman's CC: {spearman_corr:.3f}")

plt.figure(figsize=(12, 6))
plt.scatter(x, y, alpha=0.6, s=30, color='blue', label='Data points with noise')
plt.plot(x, y_clean, 'r-', linewidth=2, label='y = 2x + 1 (clean)')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title(f'Linear Relationship with Noise: y = 2x + 1 + N(0, {noise_level})\nXi: {xi_scalar:.3f}, Chatterjee\'s CC: {cc:.3f}, Normalized CC: {norm_cc:.3f}, M-NN CC: {ccc_mnn:.3f}\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 