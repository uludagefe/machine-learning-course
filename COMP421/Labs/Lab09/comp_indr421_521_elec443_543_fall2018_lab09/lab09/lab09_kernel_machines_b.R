# read data into memory
data_set <- read.csv("lab09_description_data_set.csv")

# get X values
X_train <- cbind(data_set$x1, data_set$x2)

# get number of samples and number of features
N_train <- nrow(X_train)
D_train <- ncol(X_train)

# define Euclidean distance function
pdist <- function(X1, X2) {
  if (identical(X1, X2) == TRUE) {
    D <- as.matrix(dist(X1))
  }
  else {
    D <- as.matrix(dist(rbind(X1, X2)))
    D <- D[1:nrow(X1), (nrow(X1) + 1):(nrow(X1) + nrow(X2))]
  }
  return(D)
}

# define Gaussian kernel function
gaussian_kernel <- function(X1, X2, sigma) {
  D <- pdist(X1, X2)
  K <- exp(-D^2 / (2 * sigma^2))
}

# calculate Gaussian kernel
s <- 1
K_train <- gaussian_kernel(X_train, X_train, s)

# set learning parameters
C <- 0.1
epsilon <- 1e-3

# add library required to solve QP problems
library(kernlab)
result <- ipop(c = -0.5 * diag(K_train), H = K_train,
               A = rep(1, N_train), b = 1, r = 0,
               l = rep(0, N_train), u = rep(C, N_train))
alpha <- result@primal
alpha[alpha > 0 & alpha < C * epsilon] <- 0
alpha[alpha > 0 & alpha > C * (1 - epsilon)] <- C

# find R parameter
support_indices <- which(alpha != 0)
active_indices <- which(alpha != 0 & alpha < C)
R <- sqrt(alpha[support_indices] %*% K_train[support_indices, support_indices] %*% alpha[support_indices] + 
          mean(diag(K_train[active_indices, active_indices])) - 2 * mean(K_train[active_indices, support_indices] %*% alpha[support_indices]))

# evaluate discriminant function on a grid
x1_interval <- seq(from = -3, to = +3, by = 0.15)
x2_interval <- seq(from = -3, to = +3, by = 0.15)
x1_grid <- matrix(x1_interval, nrow = length(x1_interval), ncol = length(x1_interval), byrow = FALSE)
x2_grid <- matrix(x2_interval, nrow = length(x2_interval), ncol = length(x2_interval), byrow = TRUE)

K_test_train <- gaussian_kernel(cbind(as.numeric(x1_grid), as.numeric(x2_grid)), X_train, s)
K_test_test <- gaussian_kernel(cbind(as.numeric(x1_grid), as.numeric(x2_grid)), cbind(as.numeric(x1_grid), as.numeric(x2_grid)), s)

discriminant_values <- matrix(as.numeric(R^2 - alpha[support_indices] %*% K_train[support_indices, support_indices] %*% alpha[support_indices]) + 2 * K_test_train %*% alpha - diag(K_test_test), length(x1_interval), length(x2_interval))

plot(X_train[, 1], X_train[, 2], 
     type = "p", pch = 19, col = "red",
     xlim = c(-3, +3), ylim = c(-3, +3),
     xlab = "x1", ylab = "x2", las = 1)
points(X_train[support_indices, 1], X_train[support_indices, 2], cex = 2.5, lwd = 2, pch = 1)
contour(x1_interval, x2_interval, discriminant_values, levels = 0, add = TRUE, lwd = 2, drawlabels = FALSE, lty = c(1, 2, 2))

