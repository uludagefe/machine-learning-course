safelog <- function(x) {
  return (log(x + 1e-100))
}

# read data into memory
data_set <- read.csv("lab04_data_set.csv")

# get X and y values
X <- cbind(data_set$x1, data_set$x2)
y_truth <- data_set$y

# get number of classes and number of samples
K <- max(y_truth)
N <- length(y_truth)

# one-of-K-encoding
Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, y_truth)] <- 1

# define the softmax function
softmax <- function(X, W, w0) {
  scores <- cbind(X, 1) %*% rbind(W, w0)
  scores <- exp(scores - matrix(apply(scores, MARGIN = 2, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}

# define the gradient functions
gradient_W <- function(X, Y_truth, Y_predicted) {
  return (-sapply(X = 1:ncol(Y_truth), function(c) colSums(matrix(Y_truth[,c] - Y_predicted[,c], nrow = nrow(X), ncol = ncol(X), byrow = FALSE) * X)))
}

gradient_w0 <- function(Y_truth, Y_predicted) {
  return (-colSums(Y_truth - Y_predicted))
}

# set learning parameters
eta <- 0.01
epsilon <- 1e-3

# randomly initalize W and w0
set.seed(421)
W <- matrix(runif(ncol(X) * K, min = -0.01, max = 0.01), ncol(X), K)
w0 <- runif(K, min = -0.01, max = 0.01)

# learn W and w0 using gradient descent
iteration <- 1
objective_values <- c()
while (1) {
  Y_predicted <- softmax(X, W, w0)
  
  objective_values <- c(objective_values, -sum(Y_truth * safelog(Y_predicted + 1e-100)))
  
  W_old <- W
  w0_old <- w0
  
  W <- W - eta * gradient_W(X, Y_truth, Y_predicted)
  w0 <- w0 - eta * gradient_w0(Y_truth, Y_predicted)
  
  if (sqrt(sum((w0 - w0_old)^2) + sum((W - W_old)^2)) < epsilon) {
    break
  }
  
  iteration <- iteration + 1
}
print(W)
print(w0)

# plot objective function during iterations
plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# calculate confusion matrix
y_predicted <- apply(Y_predicted, 1, which.max)
confusion_matrix <- table(y_predicted, y_truth)
print(confusion_matrix)

# evaluate discriminat function on a grid
x1_interval <- seq(from = -6, to = +6, by = 0.06)
x2_interval <- seq(from = -6, to = +6, by = 0.06)
x1_grid <- matrix(x1_interval, nrow = length(x1_interval), ncol = length(x1_interval), byrow = FALSE)
x2_grid <- matrix(x2_interval, nrow = length(x2_interval), ncol = length(x2_interval), byrow = TRUE)

discriminant_values <- array(0, c(length(x1_interval), length(x2_interval), K))
for (c in 1:K) {
  f <- function(x1, x2) {W[1, c] * x1 + W[2, c] * x2 + w0[c]}
  discriminant_values[,,c] <- exp(matrix(mapply(f, x1_grid, x2_grid), nrow(x2_grid), ncol(x2_grid)))
}

plot(X[y_truth == 1, 1], X[y_truth == 1, 2], type = "p", pch = 19, col = "red",
     xlim = c(-6, +6),
     ylim = c(-6, +6),
     xlab = "x1", ylab = "x2", las = 1)
points(X[y_truth == 2, 1], X[y_truth == 2, 2], type = "p", pch = 19, col = "green")
points(X[y_truth == 3, 1], X[y_truth == 3, 2], type = "p", pch = 19, col = "blue")
points(X[y_predicted != y_truth, 1], X[y_predicted != y_truth, 2], cex = 1.5, lwd = 2)
points(x1_grid[apply(discriminant_values, c(1, 2), which.max) == 1], x2_grid[apply(discriminant_values, c(1, 2), which.max) == 1], col = rgb(red = 1, green = 0, blue = 0, alpha = 0.05), pch = 16)
points(x1_grid[apply(discriminant_values, c(1, 2), which.max) == 2], x2_grid[apply(discriminant_values, c(1, 2), which.max) == 2], col = rgb(red = 0, green = 1, blue = 0, alpha = 0.05), pch = 16)
points(x1_grid[apply(discriminant_values, c(1, 2), which.max) == 3], x2_grid[apply(discriminant_values, c(1, 2), which.max) == 3], col = rgb(red = 0, green = 0, blue = 1, alpha = 0.05), pch = 16)
A <- discriminant_values[,,1]
B <- discriminant_values[,,2]
C <- discriminant_values[,,3]
A[A < B & A < C] <- NA
B[B < A & B < C] <- NA
C[C < A & C < B] <- NA
discriminant_values[,,1] <- A
discriminant_values[,,2] <- B
discriminant_values[,,3] <- C
contour(x1_interval, x1_interval, discriminant_values[,,1] - discriminant_values[,,2], levels = c(0), add = TRUE, lwd = 2, drawlabels = FALSE)
contour(x1_interval, x1_interval, discriminant_values[,,1] - discriminant_values[,,3], levels = c(0), add = TRUE, lwd = 2, drawlabels = FALSE)
contour(x1_interval, x1_interval, discriminant_values[,,2] - discriminant_values[,,3], levels = c(0), add = TRUE, lwd = 2, drawlabels = FALSE)
