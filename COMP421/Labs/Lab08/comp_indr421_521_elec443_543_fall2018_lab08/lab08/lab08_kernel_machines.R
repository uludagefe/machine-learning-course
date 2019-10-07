# read data into memory
data_set <- read.csv("lab08_data_set.csv")

# get X and y values
X_train <- cbind(data_set$x1, data_set$x2)
y_train <- 2 * (data_set$y == 1) - 1

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
s <- 1
K_train <- gaussian_kernel(X_train, X_train, s)
yyK <- (y_train %*% t(y_train)) * K_train

# set learning parameters
C <- 10
epsilon <- 1e-3

# add library required to solve QP problems
library(kernlab)
result <- ipop(c = rep(-1, N_train), H = yyK,
               A = y_train, b = 0, r = 0,
               l = rep(0, N_train), u = rep(C, N_train))
alpha <- result@primal
alpha[alpha < C * epsilon] <- 0
alpha[alpha > C * (1 - epsilon)] <- C

# find bias parameter
support_indices <- which(alpha != 0)
active_indices <- which(alpha != 0 & alpha < C)
b <- mean(y_train[active_indices] * (1 - yyK[active_indices, support_indices] %*% alpha[support_indices]))

# calculate predictions on training samples
f_predicted <- K_train %*% (y_train * alpha) + b

# calculate confusion matrix
y_predicted <- 2 * (f_predicted > 0) - 1
confusion_matrix <- table(y_predicted, y_train)
print(confusion_matrix)

# evaluate discriminant function on a grid
x1_interval <- seq(from = -6, to = +6, by = 0.3)
x2_interval <- seq(from = -6, to = +6, by = 0.3)
x1_grid <- matrix(x1_interval, nrow = length(x1_interval), ncol = length(x1_interval), byrow = FALSE)
x2_grid <- matrix(x2_interval, nrow = length(x2_interval), ncol = length(x2_interval), byrow = TRUE)

K_test <- gaussian_kernel(cbind(as.numeric(x1_grid), as.numeric(x2_grid)), X_train, s)
discriminant_values <- matrix(K_test %*% (y_train * alpha) + b, length(x1_interval), length(x2_interval))

plot(X_train[y_train == 1, 1], X_train[y_train == 1, 2], 
     type = "p", pch = 19, col = "red",
     xlim = c(-6, +6), ylim = c(-6, +6),
     xlab = "x1", ylab = "x2", las = 1)
points(X_train[y_train == -1, 1], X_train[y_train == -1, 2], type = "p", pch = 19, col = "blue")
points(X_train[support_indices, 1], X_train[support_indices, 2], cex = 1.5, lwd = 2)
points(x1_grid[discriminant_values > 0], x2_grid[discriminant_values > 0], col = rgb(red = 1, green = 0, blue = 0, alpha = 0.01), pch = 19, cex = 5)
points(x1_grid[discriminant_values < 0], x2_grid[discriminant_values < 0], col = rgb(red = 0, green = 0, blue = 1, alpha = 0.01), pch = 19, cex = 5)
contour(x1_interval, x2_interval, discriminant_values, levels = c(0, -1, +1), add = TRUE, lwd = 2, drawlabels = FALSE, lty = c(1, 2, 2))
