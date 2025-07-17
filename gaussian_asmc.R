library(ggplot2)
library(dplyr)

#set.seed(123)

d <- 200
N <- 200
T <- d  # intermediate distributions (following paper)
threshold <- 0.5
alpha_0 <- 0.01

# Data
A <- matrix(rnorm(d * d, 0, 1 / sqrt(d)), d, d)
x_true <- rnorm(d, mean = 2, sd = .1)
y <- A %*% x_true + rnorm(d)

# Tempering schedule
alpha_seq <- alpha_0 + (0:(d - 1)) * (1 - alpha_0) / d

# Particle storage
particles <- matrix(rnorm(N * d, sd = 3), N, d)
logw <- rep(0, N)
ESS <- numeric(T)

# Function: log target density up to constant
log_target <- function(x) {
  -0.5 * sum(x^2) - 0.5 * sum((y - A %*% x)^2)
}
 
# Store trace
mean_trace <- matrix(0, nrow = T, ncol = d)
var_trace  <- matrix(0, nrow = T, ncol = d)

# SMC loop
for (t in 2:T) {
  alpha_diff <- alpha_seq[t] - alpha_seq[t-1]
  
  # Weight update
  for (i in 1:N) {
    logw[i] <- logw[i] + alpha_diff * (-0.5 * sum((y - A %*% particles[i,])^2))
  }
  
  # Normalize log-weights
  max_logw <- max(logw)
  w <- exp(logw - max_logw)
  w <- w / sum(w)
  
  # Compute ESS
  ESS[t] <- (sum(w)^2) / sum(w^2)
  cat(sprintf("Step %d / %d - ESS = %.1f\n", t, T, ESS[t]))
  
  # Resample
  if (ESS[t] < N * threshold) {
    ancestors <- sample(1:N, N, replace = TRUE, prob = w)
    particles <- particles[ancestors, ]
    logw <- rep(0, N)
  }
  

  # Move particles (random walk)
  particles <- particles + matrix(rnorm(N * d, 0, 0.1), N, d)
  
  # Store trace
  mean_trace[t, ] <- colMeans(particles)
  var_trace[t, ]  <- apply(particles, 2, var)
}

# Posterior mean estimate
posterior_mean <- colMeans(particles)

# Compare true vs estimate
plot(x_true, posterior_mean, xlab = "True x", ylab = "Estimated posterior mean",
     main = paste("High-dim SMC (d=", d, ")", sep=""))
abline(0, 1, col = "red")

# Plot ESS over time
plot(1:T, ESS, type = "b", pch = 19,
     xlab = "SMC step", ylab = "Effective Sample Size",
     main = "ESS during SMC")


# Check
posterior_mean_weighted <- colSums(particles * w)
posterior_cov_weighted <- matrix(0, d, d)
for (i in 1:N) {
  diff <- particles[i, ] - posterior_mean_weighted
  posterior_cov_weighted <- posterior_cov_weighted + w[i] * (diff %*% t(diff))
}


# Mean-variance space
mv_data <- data.frame(
  step = rep(1:T, each = d),
  component = rep(1:d, times = T),
  mean = as.vector(mean_trace),
  var = as.vector(var_trace),
  alpha = rep(alpha_seq, each = d)
)

# Sample a subset of components for visualization clarity (optional)
# plot_components <- sample(1:d, 10)


# Plot: points only
ggplot(mv_data %>% filter(component %in% 1:d),
       aes(x = mean, y = var, color = alpha)) +
  geom_point(alpha = 0.5, size = 1.2) +
  scale_y_log10() +
  scale_color_gradient(low = "yellow", high = "blue") +
  labs(x = "Mean", y = "Variance", color = expression(alpha),
       title = "Marginal Points in Mean–Variance Space",
       subtitle = "Points represent SMC steps across components") +
  theme_minimal()

# True vs posterior mean
plot(x_true, posterior_mean_weighted, pch = 20, col = "blue",
     xlab = "True x", ylab = "Posterior Mean Estimate",
     main = "Posterior Recovery")
abline(0, 1, col = "red", lwd = 2)