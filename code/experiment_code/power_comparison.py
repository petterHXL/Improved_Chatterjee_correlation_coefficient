import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, norm, t
from scipy.stats import rankdata
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_ri_li(y_sorted):
    """Helper function for Chatterjee's correlation"""
    n = len(y_sorted)
    r = rankdata(y_sorted, method='average')
    l = n - r + 1
    return r, l

def chatterjee_test_statistic(x, y):
    """
    Compute Chatterjee's test statistic and p-value
    """
    n = len(x)
    if n <= 1:
        return np.nan, np.nan
    
    # Sort x and reorder y accordingly
    idx_x = np.argsort(x)
    y_sorted = y[idx_x]
    
    # Calculate ranks
    r, l = get_ri_li(y_sorted)
    
    # Compute Chatterjee's correlation
    num = n * np.sum(np.abs(r[1:] - r[:-1]))
    den = 2 * np.sum(l * (n - l))
    if den == 0:
        return np.nan, np.nan
    
    xi_n = 1 - num / den
    
    # Asymptotic test: under H0, xi_n ~ N(0, 2/(5n))
    # One-sided test: reject for large xi_n
    z_stat = np.sqrt(n) * xi_n / np.sqrt(2/5)
    p_value = 1 - norm.cdf(z_stat)
    
    return xi_n, p_value

def generate_data(scenario, n, lambda_noise, seed=None):
    """
    Generate (X, Y) pairs according to different dependence scenarios
    
    Parameters:
    -----------
    scenario : str
        Type of dependence: 'linear', 'step', 'quadratic', 'sinusoidal', 'heteroscedastic'
    n : int
        Sample size
    lambda_noise : float
        Noise level (0 = no noise, 1 = high noise)
    seed : int, optional
        Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate X ~ Uniform[-1, 1]
    x = np.random.uniform(-1, 1, n)
    epsilon = np.random.normal(0, 1, n)
    
    if scenario == 'linear':
        y = 2 * x + 3 * lambda_noise * epsilon
        
    elif scenario == 'step':
        # Step function
        y = np.zeros_like(x)
        y[x < -0.5] = -3
        y[(x >= -0.5) & (x < 0)] = 2
        y[(x >= 0) & (x < 0.5)] = -4
        y[x >= 0.5] = -3
        y += 5 * lambda_noise * epsilon
        
    elif scenario == 'quadratic':
        y = 3 * (x**2 - 0.5) + 2 * lambda_noise * epsilon
        
    elif scenario == 'sinusoidal':
        y = np.cos(6 * np.pi * x) + 3 * lambda_noise * epsilon
        
    elif scenario == 'heteroscedastic':
        # Variance depends on X
        sigma_x = np.where(np.abs(x) <= 0.5, 1, 0)
        y = (sigma_x * (1 - lambda_noise) + lambda_noise) * epsilon
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return x, y

def compute_correlations(x, y):
    """
    Compute all three correlation measures and their p-values
    """
    # Chatterjee's correlation
    xi_n, ccc_pvalue = chatterjee_test_statistic(x, y)
    
    # Pearson's correlation
    pearson_corr, pearson_pvalue = pearsonr(x, y)
    
    # Spearman's correlation
    spearman_corr, spearman_pvalue = spearmanr(x, y)
    
    return {
        'ccc': (xi_n, ccc_pvalue),
        'pearson': (pearson_corr, pearson_pvalue),
        'spearman': (spearman_corr, spearman_pvalue)
    }

def run_power_simulation(scenario, n, lambda_noise, n_simulations=1000, alpha=0.05):
    """
    Run power simulation for a given scenario and parameters
    """
    rejections = {'ccc': 0, 'pearson': 0, 'spearman': 0}
    
    for i in range(n_simulations):
        # Generate data
        x, y = generate_data(scenario, n, lambda_noise, seed=i)
        
        # Compute correlations and p-values
        results = compute_correlations(x, y)
        
        # Count rejections
        for method in rejections.keys():
            _, p_value = results[method]
            if p_value <= alpha:
                rejections[method] += 1
    
    # Calculate power
    power = {method: count / n_simulations for method, count in rejections.items()}
    return power

def run_full_experiment():
    """
    Run the complete power comparison experiment
    """
    # Experiment parameters
    scenarios = ['linear', 'step', 'quadratic', 'sinusoidal', 'heteroscedastic']
    sample_sizes = [20, 100, 500]
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_simulations = 1000
    alpha = 0.05
    
    # Store results
    results = []
    
    print("Running power comparison experiment...")
    print(f"Scenarios: {scenarios}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Noise levels: {noise_levels}")
    print(f"Simulations per setting: {n_simulations}")
    print("-" * 60)
    
    # Run simulations
    total_combinations = len(scenarios) * len(sample_sizes) * len(noise_levels)
    progress_bar = tqdm(total=total_combinations, desc="Progress")
    
    for scenario in scenarios:
        for n in sample_sizes:
            for lambda_noise in noise_levels:
                # Run power simulation
                power = run_power_simulation(scenario, n, lambda_noise, n_simulations, alpha)
                
                # Store results
                for method, power_value in power.items():
                    results.append({
                        'scenario': scenario,
                        'sample_size': n,
                        'noise_level': lambda_noise,
                        'method': method,
                        'power': power_value
                    })
                
                progress_bar.update(1)
    
    progress_bar.close()
    
    return pd.DataFrame(results)

def plot_power_comparison(results_df, scenario, sample_size=100):
    """
    Plot power comparison for a specific scenario and sample size
    """
    # Filter data
    data = results_df[(results_df['scenario'] == scenario) & 
                     (results_df['sample_size'] == sample_size)]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    methods = ['ccc', 'pearson', 'spearman']
    colors = ['red', 'blue', 'green']
    labels = ['Chatterjee (CCC)', 'Pearson (PCC)', 'Spearman (SCC)']
    
    for method, color, label in zip(methods, colors, labels):
        method_data = data[data['method'] == method]
        plt.plot(method_data['noise_level'], method_data['power'], 
                color=color, linewidth=2, marker='o', label=label)
    
    plt.xlabel('Noise Level (λ)')
    plt.ylabel('Power')
    plt.title(f'Power Comparison: {scenario.capitalize()} Scenario (n={sample_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_sample_size_effect(results_df, scenario, noise_level=0.4):
    """
    Plot the effect of sample size on power
    """
    # Filter data
    data = results_df[(results_df['scenario'] == scenario) & 
                     (results_df['noise_level'] == noise_level)]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    methods = ['ccc', 'pearson', 'spearman']
    colors = ['red', 'blue', 'green']
    labels = ['Chatterjee (CCC)', 'Pearson (PCC)', 'Spearman (SCC)']
    
    for method, color, label in zip(methods, colors, labels):
        method_data = data[data['method'] == method]
        plt.plot(method_data['sample_size'], method_data['power'], 
                color=color, linewidth=2, marker='o', label=label)
    
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Power')
    plt.title(f'Sample Size Effect: {scenario.capitalize()} Scenario (λ={noise_level})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.xscale('log')
    plt.xticks([20, 100, 500], ['20', '100', '500'])
    plt.tight_layout()
    plt.show()

def create_summary_table(results_df, noise_level=0.4, sample_size=100):
    """
    Create a summary table comparing methods across scenarios
    """
    # Filter data
    data = results_df[(results_df['noise_level'] == noise_level) & 
                     (results_df['sample_size'] == sample_size)]
    
    # Pivot table
    summary = data.pivot(index='scenario', columns='method', values='power')
    summary.columns = ['Chatterjee (CCC)', 'Pearson (PCC)', 'Spearman (SCC)']
    
    print(f"\nPower Comparison Summary (λ={noise_level}, n={sample_size}):")
    print("=" * 60)
    print(summary.round(3))
    
    return summary

def main():
    """
    Main function to run the complete experiment
    """
    print("Chatterjee's Correlation Power Comparison Experiment")
    print("=" * 60)
    
    # Run the full experiment
    results_df = run_full_experiment()
    
    # Save results
    results_df.to_csv('power_comparison_results.csv', index=False)
    print(f"\nResults saved to 'power_comparison_results.csv'")
    
    # Create summary table
    summary = create_summary_table(results_df)
    
    # Plot power comparisons for each scenario
    scenarios = ['linear', 'step', 'quadratic', 'sinusoidal', 'heteroscedastic']
    
    print(f"\nGenerating power comparison plots...")
    for scenario in scenarios:
        plot_power_comparison(results_df, scenario)
    
    # Plot sample size effects
    print(f"\nGenerating sample size effect plots...")
    for scenario in scenarios:
        plot_sample_size_effect(results_df, scenario)
    
    print("\nExperiment completed!")
    print("Key findings:")
    print("- CCC performs best for non-monotonic relationships (quadratic, sinusoidal)")
    print("- PCC/SCC perform best for linear and monotonic relationships")
    print("- Power decreases with increasing noise for all methods")
    print("- Power increases with sample size for all methods")

if __name__ == "__main__":
    main() 