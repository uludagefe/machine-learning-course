safelog <- function(x) {
  x[x == 0] <- 1
  return (log(x))
}

# read data into memory
data_set <- read.csv("lab03_data_set.csv")

# get X and y values
X <- cbind(data_set$x1, data_set$x2)
y_truth <- data_set$y

# get number of samples
N <- length(y_truth)

# define the sigmoid function
sigmoid <- function(X, w, w0) {
  return (1 / (1 + exp(-(X %*% w + w0))))
}

# define the gradient functions
gradient_w <- function(X, y_truth, y_predicted) {
  return (-colSums(matrix(y_truth - y_predicted, nrow = nrow(X), ncol = ncol(X), byrow = FALSE) * X))
}

gradient_w0 <- function(y_truth, y_predicted) {
  return (-sum(y_truth - y_predicted))
}

# set learning parameters
eta <- 0.01
epsilon <- 1e-3

# randomly initalize w and w0
set.seed(421)
w <- runif(ncol(X), min = -0.01, max = 0.01)
w0 <- runif(1, min = -0.01, max = 0.01)

# learn w and w0 using gradient descent
iteration <- 1
objective_values <- c()
while (1) {
  y_predicted <- sigmoid(X, w, w0)
  
  objective_values <- c(objective_values, -sum(y_truth * safelog(y_predicted) + (1 - y_truth) * safelog(1 - y_predicted)))
  
  w_old <- w
  w0_old <- w0
  
  w <- w - eta * gradient_w(X, y_truth, y_predicted)
  w0 <- w0 - eta * gradient_w0(y_truth, y_predicted)
  
  if (sqrt((w0 - w0_old)^2 + sum((w - w_old)^2)) < epsilon) {
    break
  }
  
  iteration <- iteration + 1
}
print(c(w, w0))

# plot objective function during iterations
plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# calculate confusion matrix
y_predicted <- 1 * (y_predicted > 0.5)
confusion_matrix <- table(y_predicted, y_truth)
print(confusion_matrix)

# evaluate discriminant function on a grid
x1_interval <- seq(from = -6, to = +6, by = 0.06)
x2_interval <- seq(from = -6, to = +6, by = 0.06)
x1_grid <- matrix(x1_interval, nrow = length(x1_interval), ncol = length(x1_interval), byrow = FALSE)
x2_grid <- matrix(x2_interval, nrow = length(x2_interval), ncol = length(x2_interval), byrow = TRUE)

f <- function(x1, x2) {w[1] * x1 + w[2] * x2 + w0}
discriminant_values <- matrix(mapply(f, x1_grid, x2_grid), nrow(x2_grid), ncol(x2_grid))

plot(X[y_truth == 1, 1], X[y_truth == 1, 2], type = "p", pch = 19, col = "red",
     xlim = c(-6, +6),
     ylim = c(-6, +6),
     xlab = "x1", ylab = "x2", las = 1)
points(X[y_truth == 0, 1], X[y_truth == 0, 2], type = "p", pch = 19, col = "blue")
points(X[y_predicted != y_truth, 1], X[y_predicted != y_truth, 2], cex = 1.5, lwd = 2)
points(x1_grid[discriminant_values > 0], x2_grid[discriminant_values > 0], col = rgb(red = 1, green = 0, blue = 0, alpha = 0.01), pch = 16)
points(x1_grid[discriminant_values < 0], x2_grid[discriminant_values < 0], col = rgb(red = 0, green = 0, blue = 1, alpha = 0.01), pch = 16)
contour(x1_interval, x2_interval, discriminant_values, levels = c(0), add = TRUE, lwd = 2, drawlabels = FALSE)
