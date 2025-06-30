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

def find_m_right_neighbors_paper(x, i, M):
    """
    Faithfully implements the paper's definition of the m-th right nearest neighbor j_m(i):
    For each m in 1..M, find the index j such that exactly m values x_k satisfy x_i < x_k <= x_j.
    If not enough right neighbors exist, set j_m(i) = i.
    Returns a list of length M.
    """
    n = len(x)
    neighbors = []
    x_i = x[i]
    # Get all indices j > i with x[j] > x_i
    right_indices = [(j, x[j]) for j in range(i+1, n) if x[j] > x_i]
    # Sort by x[j], then by index for deterministic tie-breaking
    right_indices.sort(key=lambda t: (t[1], t[0]))
    # For each m, find the j_m(i) as per the paper
    for m in range(1, M+1):
        if len(right_indices) < m:
            neighbors.append(i)  # fallback to i
        else:
            # j_m(i) is the index j such that exactly m values x_k satisfy x_i < x_k <= x_j
            # Since right_indices is sorted, the m-th element gives us the correct j
            neighbors.append(right_indices[m-1][0])
    return neighbors

def chatterjee_cc_mnn_with_ties(x, y, M):
    """
    Faithful implementation of the M-NN Chatterjee's correlation coefficient (Eq. 2.4 in the paper).
    """
    n = len(x)
    if n <= 1 or M <= 0:
        return np.nan
    y_ranks = rankdata(y, method='average')
    total_sum = 0
    for i in range(n):
        neighbors = find_m_right_neighbors_paper(x, i, M)
        for j in neighbors:
            total_sum += min(y_ranks[i], y_ranks[j])
    denominator = (n+1) * (n*M + M*(M+1)/4)
    if denominator == 0:
        return np.nan
    return -2 + (6 * total_sum) / denominator 

def normalized_chatterjee_cc_mnn(x, y, M):
    """
    Compute the normalized M-NN Chatterjee's correlation coefficient.
    Normalization is performed as:
        xi'_M = max(-1, xi_M(x, y) / xi_M(y, y))
    where xi_M(x, y) is the M-NN Chatterjee's correlation coefficient between x and y,
    and xi_M(y, y) is the maximum possible value (when x = y).
    This ensures the normalized value is in [-1, 1] and comparable across datasets.
    """
    raw_xi_mnn = chatterjee_cc_mnn_with_ties(x, y, M)
    max_possible = chatterjee_cc_mnn_with_ties(y, y, M)
    if max_possible == 0:
        return np.nan
    xi_prime_mnn = raw_xi_mnn / max_possible
    return max(-1, xi_prime_mnn) 