# read data into memory
data_set <- read.csv("lab06_data_set.csv")

# get x and y values
x_train <- data_set$x
y_train <- data_set$y

# get number of classes and number of samples
K <- max(y_train)
N <- length(y_train)

point_colors <- c("red", "green", "blue")
minimum_value <- -8
maximum_value <- +8
data_interval <- seq(from = minimum_value, to = maximum_value, by = 0.01)

# histogram estimator
bin_width <- 0.5
left_borders <- seq(from = minimum_value, to = maximum_value - bin_width, by = bin_width)
right_borders <- seq(from = minimum_value + bin_width, to = maximum_value, by = bin_width)
p_head <- sapply(1:length(left_borders), function(b) {sum(left_borders[b] < x_train & x_train <= right_borders[b])}) / (N * bin_width)

plot(x_train, -0.01 * y_train, type = "p", pch = 19, col = point_colors[y_train],
     ylim = c(-0.03, max(p_head)), xlim = c(minimum_value, maximum_value),
     ylab = "density", xlab = "x", las = 1, main = sprintf("h = %g", bin_width))
for (b in 1:length(left_borders)) {
  lines(c(left_borders[b], right_borders[b]), c(p_head[b], p_head[b]), lwd = 2, col = "black")
  if (b < length(left_borders)) {
    lines(c(right_borders[b], right_borders[b]), c(p_head[b], p_head[b + 1]), lwd = 2, col = "black") 
  }
}

# naive estimator
bin_width <- 0.5
p_head <- sapply(data_interval, function(x) {sum((x - 0.5 * bin_width) < x_train & x_train <= (x + 0.5 * bin_width))}) / (N * bin_width)

plot(x_train, -0.01 * y_train, type = "p", pch = 19, col = point_colors[y_train],
     ylim = c(-0.03, max(p_head)), xlim = c(minimum_value, maximum_value),
     ylab = "density", xlab = "x", las = 1, main = sprintf("h = %g", bin_width))
lines(data_interval, p_head, type = "l", lwd = 2, col = "black")

# kernel estimator
bin_width <- 0.5
p_head <- sapply(data_interval, function(x) {sum(1 / sqrt(2 * pi) * exp(-0.5 * (x - x_train)^2 / bin_width^2))}) / (N * bin_width)

plot(x_train, -0.01 * y_train, type = "p", pch = 19, col = point_colors[y_train],
     ylim = c(-0.03, max(p_head)), xlim = c(minimum_value, maximum_value),
     ylab = "density", xlab = "x", las = 1, main = sprintf("h = %g", bin_width))
lines(data_interval, p_head, type = "l", lwd = 2, col = "black")

# k-nearest neighbor estimator
k <- 11
p_head <- sapply(data_interval, function(x) {k / (2 * N * sort(abs(x - x_train), decreasing = FALSE)[k])})

plot(x_train, -0.01 * y_train, type = "p", pch = 19, col = point_colors[y_train],
     ylim = c(-0.03, max(p_head)), xlim = c(minimum_value, maximum_value),
     ylab = "density", xlab = "x", las = 1, main = sprintf("k = %g", k))
lines(data_interval, p_head, type = "l", lwd = 2, col = "black")

# nonparametric classification
bin_width <- 0.5
p_head <- matrix(0, length(data_interval), K)
for (c in 1:K) {
  p_head[,c] <- sapply(data_interval, function(x) {sum(1 / sqrt(2 * pi) * exp(-0.5 * (x - x_train[y_train == c])^2 / bin_width^2))}) / (N * bin_width) 
}

plot(x_train, -0.01 * y_train, type = "p", pch = 19, col = point_colors[y_train],
     ylim = c(-0.03, max(p_head)), xlim = c(minimum_value, maximum_value),
     ylab = "density", xlab = "x", las = 1, main = sprintf("h = %g", bin_width))
for (c in 1:K) {
  lines(data_interval, p_head[,c], type = "l", lwd = 2, col = point_colors[c]) 
}
points(data_interval, rep(max(p_head), length(data_interval)),
       col = point_colors[apply(p_head, 1, which.max)], pch = 19)

# knn classification
k <- 11
p_head <- matrix(0, length(data_interval), K)
for (c in 1:K) {
  p_head[,c] <- sapply(data_interval, function(x) {sum(y_train[order(abs(x - x_train), decreasing = FALSE)[1:k]] == c) / k}) 
}

plot(x_train, -0.01 * y_train, type = "p", pch = 19, col = point_colors[y_train],
     ylim = c(-0.03, max(p_head)), xlim = c(minimum_value, maximum_value),
     ylab = "density", xlab = "x", las = 1, main = sprintf("k = %g", k))
for (c in 1:K) {
  lines(data_interval, p_head[,c], type = "l", lwd = 2, col = point_colors[c]) 
}
points(data_interval, rep(max(p_head), length(data_interval)),
       col = point_colors[apply(p_head, 1, which.max)], pch = 19)