# read data into memory
training_digits <- read.csv("lab10_mnist_training_digits.csv", header = FALSE)
training_labels <- read.csv("lab10_mnist_training_labels.csv", header = FALSE)

# get X and y values
X <- as.matrix(training_digits) / 255
y <- training_labels[,1]

# get number of samples and number of features
N <- length(y)
D <- ncol(X)

# calculate the covariance matrix
Sigma_X <- cov(X)

# calculate the eigenvalues and eigenvectors
decomposition <- eigen(Sigma_X, symmetric = TRUE)

# plot scree graph
plot(1:D, decomposition$values, 
     type = "l", las = 1, lwd = 2,
     xlab = "Eigenvalue index", ylab = "Eigenvalue")

# plot proportion of variance explained
pove <- cumsum(decomposition$values) / sum(decomposition$values)
plot(1:D, pove, 
     type = "l", las = 1, lwd = 2,
     xlab = "R", ylab = "Proportion of variance explained")
abline(h = 0.95, lwd = 2, lty = 2, col = "blue")
abline(v = which(pove > 0.95)[1], lwd = 2, lty = 2, col = "blue")

# calculate two-dimensional projections
Z <- (X - matrix(colMeans(X), N, D, byrow = TRUE)) %*% decomposition$vectors[,1:2]

# plot two-dimensional projections
point_colors <- c("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6")
plot(Z[,1], Z[,2], type = "p", pch = 19, col = point_colors[y], cex = 0,
     xlab = "PC1", ylab = "PC2", las = 1)
text(Z[,1], Z[,2], labels = y %% 10, col = point_colors[y])

# calculate reconstruction error
reconstruction_error <- c()
for (r in 1:min(D, 100)) {
  Z_r <- (X - matrix(colMeans(X), N, D, byrow = TRUE)) %*% decomposition$vectors[,1:r]
  X_reconstructed <- Z_r %*% t(decomposition$vectors[,1:r]) + matrix(colMeans(X), N, D, byrow = TRUE)
  reconstruction_error[r] <- mean((X - X_reconstructed)^2)
}

plot(1:min(D, 100), reconstruction_error[1:min(D, 100)], 
     type = "l", las = 1, lwd = 2,
     xlab = "R", ylab = "Average reconstruction error")
abline(h = 0, lwd = 2, lty = 2, col = "blue")

# plot first 100 eigenvectors
layout(matrix(1:100, 10, 10, byrow = TRUE))
par(mar = c(0, 0, 0, 0), oma = c(0, 0, 0, 0))
for (component in 1:100) {
  image(matrix(decomposition$vectors[,component], nrow = 28)[,28:1], col = gray(12:1/12), axes = FALSE)
  dev.next()
}
