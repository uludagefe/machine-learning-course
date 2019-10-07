safelog <- function(x) {
  return (log(x + 1e-100))
}

# read data into memory
data_set <- read.csv("lab05_data_set.csv")

# get X and y values
X <- cbind(data_set$x1, data_set$x2)
y_truth <- data_set$y

# get number of samples and number of features
N <- length(y_truth)
D <- ncol(X)

# define the sigmoid function
sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

# set learning parameters
eta <- 0.1
epsilon <- 1e-3
H <- 20
max_iteration <- 200

# randomly initalize W and v
set.seed(421)
W <- matrix(runif((D + 1) * H, min = -0.01, max = 0.01), D + 1, H)
v <- runif(H + 1, min = -0.01, max = 0.01)

Z <- sigmoid(cbind(1, X) %*% W)
y_predicted <- sigmoid(cbind(1, Z) %*% v)
objective_values <- -sum(y_truth * safelog(y_predicted) + (1 - y_truth) * safelog(1 - y_predicted))

# learn W and v using gradient descent and online learning
iteration <- 1
while (1) {
  for (i in sample(N)) {
    # calculate hidden nodes
    Z[i,] <- sigmoid(c(1, X[i,]) %*% W)
    # calculate output node
    y_predicted[i] <- sigmoid(c(1, Z[i,]) %*% v)

    v <- v + eta * (y_truth[i] - y_predicted[i]) * c(1, Z[i,])
    for (h in 1:H) {
      W[,h] <- W[,h] + eta * (y_truth[i] - y_predicted[i]) * v[h + 1] * Z[i, h] * (1 - Z[i, h]) * c(1, X[i,])
    }
  }

  Z <- sigmoid(cbind(1, X) %*% W)
  y_predicted <- sigmoid(cbind(1, Z) %*% v)
  objective_values <- c(objective_values, -sum(y_truth * safelog(y_predicted) + (1 - y_truth) * safelog(1 - y_predicted)))
  
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
}
print(W)
print(v)

# plot objective function during iterations
plot(1:(iteration + 1), objective_values,
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

f <- function(x1, x2) { c(1, sigmoid(c(1, x1, x2) %*% W)) %*% v }
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

