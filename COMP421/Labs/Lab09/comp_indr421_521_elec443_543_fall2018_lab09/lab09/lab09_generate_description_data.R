library(MASS)

set.seed(421)
# mean parameters
class_means <- matrix(c(+2.0, +2.0,
                        -1.5, +0.5,
                        +1.5, -1.5), 2, 3)
# covariance parameters
class_covariances <- array(c(+0.1, 0.0, 0.0, +0.1,
                             +0.2, 0.0, 0.0, +0.2,
                             +0.2, 0.0, 0.0, +0.2), c(2, 2, 3))
# sample sizes
class_sizes <- c(2, 49, 49)

# generate random samples
points1 <- mvrnorm(n = class_sizes[1], mu = class_means[,1], Sigma = class_covariances[,,1])
points2 <- mvrnorm(n = class_sizes[2], mu = class_means[,2], Sigma = class_covariances[,,2])
points3 <- mvrnorm(n = class_sizes[3], mu = class_means[,3], Sigma = class_covariances[,,3])
X <- rbind(points1, points2, points3)
colnames(X) <- c("x1", "x2")

# write data to a file
write.csv(x = X, file = "lab09_description_data_set.csv", row.names = FALSE)

# plot data points generated
plot(X[,1], X[,2], type = "p", pch = 19, col = "black", las = 1,
     xlim = c(-3, 3), ylim = c(-3, 3),
     xlab = "x1", ylab = "x2")
