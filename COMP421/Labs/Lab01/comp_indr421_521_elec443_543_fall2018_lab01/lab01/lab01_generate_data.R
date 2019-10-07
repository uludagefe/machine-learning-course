set.seed(421)
# mean parameters
class_means <- c(-3, -1, 3)
# standard deviation parameters
class_deviations <- c(1.2, 1.0, 1.3)
# sample sizes
class_sizes <- c(40, 30, 50)

# generate random samples
points1 <- rnorm(n = class_sizes[1], mean = class_means[1], sd = class_deviations[1])
points2 <- rnorm(n = class_sizes[2], mean = class_means[2], sd = class_deviations[2])
points3 <- rnorm(n = class_sizes[3], mean = class_means[3], sd = class_deviations[3])
x <- c(points1, points2, points3)

# generate corresponding labels
y <- c(rep(1, class_sizes[1]), rep(2, class_sizes[2]), rep(3, class_sizes[3]))

# write data to a file
write.csv(x = cbind(x, y), file = "lab01_data_set.csv", row.names = FALSE)

# plot densities used and data points generated together
data_interval <- seq(from = -7, to = +7, by = 0.01)
density1 <- dnorm(data_interval, mean = class_means[1], sd = class_deviations[1])
density2 <- dnorm(data_interval, mean = class_means[2], sd = class_deviations[2])
density3 <- dnorm(data_interval, mean = class_means[3], sd = class_deviations[3])
plot(data_interval, density1, type = "l", col = "red", lwd = 2, 
     ylim = c(-0.03, max(density1, density2, density3)),
     xlab = "x", ylab = "density", las = 1)
points(data_interval, density2, type = "l", col = "green", lwd = 2)
points(data_interval, density3, type = "l", col = "blue", lwd = 2)
points(points1, rep(-0.01, class_sizes[1]), type = "p", pch = 19, col = "red")
points(points2, rep(-0.02, class_sizes[2]), type = "p", pch = 19, col = "green")
points(points3, rep(-0.03, class_sizes[3]), type = "p", pch = 19, col = "blue")
