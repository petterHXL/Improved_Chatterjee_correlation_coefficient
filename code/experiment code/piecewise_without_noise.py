import numpy as np
import xicorpy as xicor
import matplotlib.pyplot as plt
from scipy.stats import rankdata, pearsonr, spearmanr

def get_ri_li(y_sorted):
    n = len(y_sorted)
    r = rankdata(y_sorted, method='average')          # r_i: how many ≤ Y_i
    l = n - r + 1                                     # l_i: how many ≥ Y_i
    return r, l

def chatterjee_cc(x, y):
    """
    Compute Chatterjee's rank correlation coefficient using the formula:
    xi_n(X, Y) = 1 - [n * sum_{i=1}^{n-1} |r_{i+1} - r_i|] / [2 * sum_{i=1}^n l_i (n - l_i)]
    where r_i is the rank of Y_i after sorting X (ascending),
    and l_i is the rank of Y_i after sorting X (descending).
    """
    n = len(x)
    if n <= 1:
        return np.nan
    
    # Sort x and reorder y accordingly
    idx_x = np.argsort(x)
    y_sorted = y[idx_x]
    
    # Use the new get_ri_li function to calculate r and l
    r, l = get_ri_li(y_sorted)
    
    num = n * np.sum(np.abs(r[1:] - r[:-1]))
    den = 2 * np.sum(l * (n - l))
    if den == 0:
        return np.nan
    return 1 - num / den

def normalized_chatterjee_cc(x, y):
    """
    Compute normalized Chatterjee's correlation as:
    xi' = xi(x, y) / xi(y, y), capped at [-1, 1]
    """
    raw_xi = chatterjee_cc(x, y)
    max_possible = chatterjee_cc(y, y)
    
    if max_possible == 0:
        return np.nan  # or 0, depending on your use case
    
    xi_prime = raw_xi / max_possible
    return max(-1, min(1, xi_prime))  # ensure bounded within [-1, 1]

def piecewise_function(x):
    """
    Create a monotonic function with upward jumps:
    Start with y = x³, then add upward shifts at specific points
    """
    y = x**3  # Start with y = x³ (monotonic for all x)
    
    # Add upward jumps at specific points
    y[x >= -2] += 50  # Jump up by 50 at x = -2
    y[x >= 0] += 50   # Jump up by 50 at x = 0  
    y[x >= 2] += 50   # Jump up by 50 at x = 2
    
    return y

# Piecewise relationship
x = np.linspace(-5, 5, 100)
y = piecewise_function(x)

# Compute all correlations
xi_result = xicor.compute_xi_correlation(x, y, get_p_values=False)
if isinstance(xi_result, tuple):
    xi = xi_result[0]  # Extract correlation value from tuple
else:
    xi = xi_result
cc = chatterjee_cc(x, y)
norm_cc = normalized_chatterjee_cc(x, y)

# Calculate Pearson's and Spearman's correlations
pearson_corr, pearson_p = pearsonr(x, y)
spearman_corr, spearman_p = spearmanr(x, y)

# Extract scalar value if xi is an array
if hasattr(xi, 'item'):
    xi_scalar = xi.item()
else:
    xi_scalar = float(xi)

print(f"Xi correlation: {xi_scalar:.3f}")
print(f"Chatterjee's CC: {cc:.3f}")
print(f"Normalized Chatterjee's CC: {norm_cc:.3f}")
print(f"Pearson's CC: {pearson_corr:.3f}")
print(f"Spearman's CC: {spearman_corr:.3f}")

# Plot the data
plt.figure(figsize=(12, 6))
plt.scatter(x, y, alpha=0.6, s=30, color='blue', label='Data points')
plt.plot(x, y, 'r-', linewidth=2, label='Piecewise function')

# Add vertical lines to show the piecewise boundaries
plt.axvline(x=-2, color='green', linestyle='--', alpha=0.7, label='Boundary at x=-2')
plt.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='Boundary at x=0')
plt.axvline(x=2, color='green', linestyle='--', alpha=0.7, label='Boundary at x=2')

plt.xlabel('X values')
plt.ylabel('Y values')
plt.title(f'Piecewise Monotonic Function with Jumps\nXi: {xi_scalar:.3f}, Chatterjee\'s CC: {cc:.3f}, Normalized CC: {norm_cc:.3f}\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 