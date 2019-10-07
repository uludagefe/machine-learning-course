# read data into memory
data_set <- read.csv("lab01_data_set.csv")

# get x and y values
x <- data_set$x
y <- data_set$y

# get number of classes and number of samples
K <- max(y)
N <- length(y)

# calculate sample means
sample_means <- sapply(X = 1:K, FUN = function(c) {mean(x[y == c])})

# calculate sample deviations
sample_deviations <- sapply(X = 1:K, FUN = function(c) {sqrt(mean((x[y == c] - sample_means[c])^2))})

# calculate prior probabilities
class_priors <- sapply(X = 1:K, FUN = function(c) {mean(y == c)})

data_interval <- seq(from = -7, to = +7, by = 0.01)

# evaluate score functions
score_values <- sapply(X = 1:K, FUN = function(c) {- 0.5 * log(2 * pi * sample_deviations[c]^2) - 0.5 * (data_interval - sample_means[c])^2 / sample_deviations[c]^2 + log(class_priors[c])})

# calculate log posteriors
log_posteriors <- score_values - sapply(X = 1:nrow(score_values), FUN = function(r) {max(score_values[r,]) + log(sum(exp(score_values[r,] - max(score_values[r,]))))})

# plot score functions
plot(data_interval, score_values[,1], type = "l", col = "red", lwd = 2, 
     ylim = c(min(score_values), 0), 
     xlab = "x", ylab = "score", las = 1)
points(data_interval, score_values[,2], type = "l", col = "green", lwd = 2)
points(data_interval, score_values[,3], type = "l", col = "blue", lwd = 2)

# plot posteriors
plot(data_interval, exp(log_posteriors[,1]), type = "l", col = "red", lwd = 2,
     ylim = c(-0.15, 1), las = 1,
     xlab = "x", ylab = "probability")
points(data_interval, exp(log_posteriors[,2]), type = "l", col = "green", lwd = 2)
points(data_interval, exp(log_posteriors[,3]), type = "l", col = "blue", lwd = 2)

class_assignments <- apply(X = score_values, MARGIN = 1, FUN = which.max)
points(data_interval[class_assignments == 1], 
       rep(-0.05, sum(class_assignments == 1)), type = "p", pch = 19, col = "red")
points(data_interval[class_assignments == 2], 
       rep(-0.10, sum(class_assignments == 2)), type = "p", pch = 19, col = "green")
points(data_interval[class_assignments == 3], 
       rep(-0.15, sum(class_assignments == 3)), type = "p", pch = 19, col = "blue")
