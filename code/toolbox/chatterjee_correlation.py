import numpy as np
from scipy.stats import rankdata

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
    
    Parameters:
    -----------
    x : array-like
        First variable
    y : array-like
        Second variable
        
    Returns:
    --------
    float
        Chatterjee's correlation coefficient value
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
    xi' = max{-1, xi(x, y) / xi(y, y)}
    
    Parameters:
    -----------
    x : array-like
        First variable
    y : array-like
        Second variable
        
    Returns:
    --------
    float
        Normalized Chatterjee's correlation coefficient value
    """
    raw_xi = chatterjee_cc(x, y)
    max_possible = chatterjee_cc(y, y)
    
    if max_possible == 0:
        return np.nan  # or 0, depending on your use case
    
    xi_prime = raw_xi / max_possible
    return max(-1, xi_prime)  # Take maximum of ratio and -1 





def find_m_right_neighbors_robust(x, y, i, M):
    """
    Find M right nearest neighbors for point i, handling ties
    """
    n = len(x)
    candidates = []
    
    for j in range(i+1, n):
        # Calculate distance metric (X-distance with Y-tie breaking)
        distance = x[j] - x[i]  # X-distance
        if y[i] == y[j]:
            distance += 1e-10 * (j - i)  # Small tie-breaker based on position
        
        candidates.append((j, distance))
    
    # Sort by distance and take top M
    candidates.sort(key=lambda x: x[1])
    return [j for j, _ in candidates[:M]]

def chatterjee_cc_mnn_with_ties(x, y, M):
    """
    M-NN CCC with robust tie handling
    """
    n = len(x)
    if n <= 1 or M <= 0:
        return np.nan
    
    # Calculate ranks for Y values
    y_ranks = rankdata(y, method='average')
    
    total_sum = 0
    for i in range(n):
        # Find M right neighbors, handling ties
        neighbors = find_m_right_neighbors_robust(x, y, i, M)
        
        if neighbors:
            # Calculate min ranks across M neighbors
            min_rank = min(y_ranks[j] for j in neighbors)
            total_sum += min_rank
        else:
            # No right neighbors available
            total_sum += y_ranks[i]
    
    # Apply M-NN formula
    denominator = (n+1) * (n*M + M*(M+1)/4)
    if denominator == 0:
        return np.nan
    
    return -2 + (6 * total_sum) / denominator 