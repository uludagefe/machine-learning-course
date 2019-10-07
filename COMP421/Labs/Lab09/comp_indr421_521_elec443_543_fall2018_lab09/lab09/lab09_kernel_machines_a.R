# read data into memory
data_set <- read.csv("lab09_regression_data_set.csv")

# get X and y values
set.seed(421)
train_indices <- sample(1:nrow(data_set), 100)
X_train <- data_set[train_indices, "x", drop = FALSE]
y_train <- data_set$y[train_indices]
X_test <- data_set[-train_indices, "x", drop = FALSE]
y_test <- data_set$y[-train_indices]

# get number of samples and number of features
N_train <- length(y_train)
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
s <- 6
K_train <- gaussian_kernel(X_train, X_train, s)

# set learning parameters
C <- 1000
tube <- 10
epsilon <- 1e-3

# add library required to solve QP problems
library(kernlab)
result <- ipop(c = c(tube - y_train, tube + y_train), H = rbind(cbind(+K_train, -K_train), cbind(-K_train, +K_train)),
               A = c(rep(1, N_train), rep(-1, N_train)), b = 0, r = 0,
               l = rep(0, 2 * N_train), u = rep(C, 2 * N_train))
alpha <- result@primal[1:N_train] - result@primal[(N_train + 1):(2 * N_train)]
alpha[alpha > 0 & alpha < +C * epsilon] <- 0
alpha[alpha < 0 & alpha > -C * epsilon] <- 0
alpha[alpha > 0 & alpha > +C * (1 - epsilon)] <- +C
alpha[alpha < 0 & alpha < -C * (1 - epsilon)] <- -C

# find bias parameter
support_indices <- which(alpha != 0)
active_indices <- which(alpha != 0 & abs(alpha) < C)
b <- mean(y_train[active_indices] - tube * sign(alpha[active_indices])) - mean(K_train[active_indices, support_indices] %*% alpha[support_indices])

# calculate predictions on training samples
y_predicted <- K_train %*% alpha + b

# calculate RMSE on training samples
rmse_train <- sqrt(mean((y_predicted - y_train)^2))
print(rmse_train)

# calculate predictions on test samples
K_test <- gaussian_kernel(X_test, X_train, s)
y_predicted <- K_test %*% alpha + b

# calculate RMSE on test samples
rmse_test <- sqrt(mean((y_predicted - y_test)^2))
print(rmse_test)

# evaluate fitted function on a grid
x_interval <- seq(from = 0, to = 60, by = 0.1)
X_interval <- data.frame(row.names = 1:length(x_interval))
X_interval$x <- x_interval

K_interval <- gaussian_kernel(X_interval, X_train, s)
fitted_values <- K_interval %*% alpha + b

plot(X_train[, 1], y_train, 
     type = "p", pch = 19, col = "black",
     xlim = c(0, 60), ylim = c(-135, +75),
     xlab = "x", ylab = "y", las = 1)
points(X_train[support_indices, 1], y_train[support_indices], cex = 2.5, lwd = 2, pch = 1)
points(X_test[, 1], y_test, type = "p", pch = 19, col = "blue")
points(X_test[, 1], y_predicted, type = "p", pch = 19, col = "red")
for (i in 1:length(y_test)) {
  lines(c(X_test[i, 1], X_test[i, 1]), c(y_test[i], y_predicted[i]), type = "l", col = "red", lty = 2, lwd = 2)
}
points(X_interval[, 1], fitted_values, type = "l", lty = 1, col = "magenta", lwd = 3)
points(X_interval[, 1], fitted_values - tube, type = "l", lty = 2, col = "magenta", lwd = 3)
points(X_interval[, 1], fitted_values + tube, type = "l", lty = 2, col = "magenta", lwd = 3)
