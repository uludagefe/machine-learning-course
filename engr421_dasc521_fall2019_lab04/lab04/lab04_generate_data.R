library(MASS)

set.seed(421)
# mean parameters
class_means <- matrix(c(+0.0, +1.5, -2.5, -3.0, +2.5, -3.0), 2, 3)
# covariance parameters
class_covariances <- array(c(+1.0, +0.2, +0.2, +3.2,
                             +1.6, -0.8, -0.8, +1.0,
                             +1.6, +0.8, +0.8, +1.0), c(2, 2, 3))
# sample sizes
class_sizes <- c(100, 100, 100)

# generate random samples
points1 <- mvrnorm(n = class_sizes[1], mu = class_means[,1], Sigma = class_covariances[,,1])
points2 <- mvrnorm(n = class_sizes[2], mu = class_means[,2], Sigma = class_covariances[,,2])
points3 <- mvrnorm(n = class_sizes[3], mu = class_means[,3], Sigma = class_covariances[,,3])
X <- rbind(points1, points2, points3)
colnames(X) <- c("x1", "x2")

# generate corresponding labels
y <- c(rep(1, class_sizes[1]), rep(2, class_sizes[2]), rep(3, class_sizes[3]))

# write data to a file
write.csv(x = cbind(X, y), file = "lab04_data_set.csv", row.names = FALSE)

# plot data points generated
plot(points1[,1], points1[,2], type = "p", pch = 19, col = "red", las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6),
     xlab = "x1", ylab = "x2")
points(points2[,1], points2[,2], type = "p", pch = 19, col = "green")
points(points3[,1], points3[,2], type = "p", pch = 19, col = "blue")
