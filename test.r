# Load required libraries
library(XICOR)
library(ggplot2)

# Function to compute Chatterjee's correlation coefficient manually
chatterjee_cc <- function(x, y) {
  n <- length(x)
  if (n <= 1) return(NA)
  
  # Sort x and reorder y accordingly
  idx_x <- order(x)
  y_sorted <- y[idx_x]
  
  # Calculate ranks
  r <- rank(y_sorted, ties.method = "average")
  l <- n - r + 1
  
  # Compute Chatterjee's correlation
  num <- n * sum(abs(diff(r)))
  den <- 2 * sum(l * (n - l))
  
  if (den == 0) return(NA)
  return(1 - num / den)
}

# Function to compute normalized Chatterjee's correlation
normalized_chatterjee_cc <- function(x, y) {
  raw_xi <- chatterjee_cc(x, y)
  max_possible <- chatterjee_cc(y, y)
  
  if (max_possible == 0) return(NA)
  
  xi_prime <- raw_xi / max_possible
  return(max(-1, min(1, xi_prime)))
}

# Linear relationship without noise
x <- seq(-5, 5, length.out = 100)
y <- 2 * x + 1  # Linear function: y = 2x + 1

# Compute all correlations
xi_result <- xicor(x, y)
cc <- chatterjee_cc(x, y)
norm_cc <- normalized_chatterjee_cc(x, y)

# Calculate Pearson's and Spearman's correlations
pearson_corr <- cor(x, y, method = "pearson")
spearman_corr <- cor(x, y, method = "spearman")

# Print results
cat("Xi correlation:", round(xi_result, 3), "\n")
cat("Chatterjee's CC:", round(cc, 3), "\n")
cat("Normalized Chatterjee's CC:", round(norm_cc, 3), "\n")
cat("Pearson's CC:", round(pearson_corr, 3), "\n")
cat("Spearman's CC:", round(spearman_corr, 3), "\n")

# Create plot
df <- data.frame(x = x, y = y)

p <- ggplot(df, aes(x = x, y = y)) +
  geom_point(alpha = 0.6, size = 2, color = "blue") +
  geom_line(color = "red", linewidth = 1) +
  labs(
    title = paste("Linear Relationship: y = 2x + 1\n",
                  "Xi:", round(xi_result, 3), 
                  "Chatterjee's CC:", round(cc, 3), 
                  "Normalized CC:", round(norm_cc, 3), "\n",
                  "Pearson:", round(pearson_corr, 3), 
                  "Spearman:", round(spearman_corr, 3)),
    x = "X values",
    y = "Y values"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10))

print(p)
