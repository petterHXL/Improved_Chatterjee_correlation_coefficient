import numpy as np
import xicorpy as xicor
import matplotlib.pyplot as plt
from scipy.stats import rankdata, pearsonr, spearmanr

# Heteroscedastic relationship: variance of Y depends on X
np.random.seed(42)
n = 100
x = np.linspace(-2, 2, n)
noise = np.random.normal(0, 1, n)
y = x * noise  # Variance increases with |x|

def get_ri_li(y_sorted):
    n = len(y_sorted)
    r = rankdata(y_sorted, method='average')
    l = n - r + 1
    return r, l

def chatterjee_cc(x, y):
    n = len(x)
    if n <= 1:
        return np.nan
    idx_x = np.argsort(x)
    y_sorted = y[idx_x]
    r, l = get_ri_li(y_sorted)
    num = n * np.sum(np.abs(r[1:] - r[:-1]))
    den = 2 * np.sum(l * (n - l))
    if den == 0:
        return np.nan
    return 1 - num / den

def normalized_chatterjee_cc(x, y):
    raw_xi = chatterjee_cc(x, y)
    max_possible = chatterjee_cc(y, y)
    if max_possible == 0:
        return np.nan
    xi_prime = raw_xi / max_possible
    return max(-1, min(1, xi_prime))

# Compute all correlations
xi_result = xicor.compute_xi_correlation(x, y, get_p_values=False)
if isinstance(xi_result, tuple):
    xi = xi_result[0]
else:
    xi = xi_result
cc = chatterjee_cc(x, y)
norm_cc = normalized_chatterjee_cc(x, y)

pearson_corr, pearson_p = pearsonr(x, y)
spearman_corr, spearman_p = spearmanr(x, y)

if hasattr(xi, 'item'):
    xi_scalar = xi.item()
else:
    xi_scalar = float(xi)

print(f"Xi correlation: {xi_scalar:.3f}")
print(f"Chatterjee's CC: {cc:.3f}")
print(f"Normalized Chatterjee's CC: {norm_cc:.3f}")
print(f"Pearson's CC: {pearson_corr:.3f}")
print(f"Spearman's CC: {spearman_corr:.3f}")

plt.figure(figsize=(12, 6))
plt.scatter(x, y, alpha=0.6, s=30, color='purple', label='Data points')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title(f'Heteroscedastic Relationship: Y = X * noise\nXi: {xi_scalar:.3f}, Chatterjee\'s CC: {cc:.3f}, Normalized CC: {norm_cc:.3f}\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 