#set.seed(123)

# Dimensionality 
d <- 50

# Number of particles and steps
N <- 1000
T <- d  # intermediate distributions
threshold <- 0.5
alpha_0 <- 0.01

# Data
A <- matrix(rnorm(d * d, 0, 1 / sqrt(d)), d, d)
x_true <- rnorm(d)
y <- A %*% x_true + rnorm(d)

# Tempering schedule
alpha_seq <- alpha_0 + (0:(d - 1)) * (1 - alpha_0) / d

# Particle storage
particles <- matrix(rnorm(N * d), N, d)
logw <- rep(0, N)
ESS <- numeric(T)

# Function: log target density up to constant
log_target <- function(x) {
  -0.5 * sum(x^2) - 0.5 * sum((y - A %*% x)^2)
}

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
  ESS[t] <- 1 / sum(w^2)
  cat(sprintf("Step %d / %d - ESS = %.1f\n", t, T, ESS[t]))
  
  # Resample
  if (ESS[t] < N * threshold) {
    ancestors <- sample(1:N, N, replace = TRUE, prob = w)
    particles <- particles[ancestors, ]
    logw <- rep(0, N)
  }
  

  # Move particles (random walk)
  particles <- particles + matrix(rnorm(N * d, 0, 0.1), N, d)
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
weighted_means <- apply(particles, 2, function(x) sum(x * w))
weighted_sds <- apply(particles, 2, function(x) sqrt(sum(w * (x - sum(x * w))^2)))

mean(weighted_means)
mean(weighted_sds)