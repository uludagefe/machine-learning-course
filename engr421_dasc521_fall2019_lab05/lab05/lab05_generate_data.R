library(MASS)

set.seed(421)
# mean parameters
class_means <- matrix(c(+2.0, +2.0,
                        -2.0, +2.0,
                        -2.0, -2.0,
                        +2.0, -2.0,
                        -4.0, -4.0,
                        +4.0, +4.0,
                        -4.0, +4.0,
                        +4.0, -4.0), 2, 8)
# covariance parameters
class_covariances <- array(c(+0.8, -0.6, -0.6, +0.8,
                             +0.8, +0.6, +0.6, +0.8,
                             +0.8, -0.6, -0.6, +0.8,
                             +0.8, +0.6, +0.6, +0.8,
                             +0.4, +0.0, +0.0, +0.4,
                             +0.4, +0.0, +0.0, +0.4,
                             +0.4, +0.0, +0.0, +0.4,
                             +0.4, +0.0, +0.0, +0.4), c(2, 2, 8))
# sample sizes
class_sizes <- c(200, 200)

# generate random samples
points1 <- mvrnorm(n = class_sizes[1] / 4, mu = class_means[,1], Sigma = class_covariances[,,1])
points2 <- mvrnorm(n = class_sizes[2] / 4, mu = class_means[,2], Sigma = class_covariances[,,2])
points3 <- mvrnorm(n = class_sizes[1] / 4, mu = class_means[,3], Sigma = class_covariances[,,3])
points4 <- mvrnorm(n = class_sizes[2] / 4, mu = class_means[,4], Sigma = class_covariances[,,4])
points5 <- mvrnorm(n = class_sizes[2] / 4, mu = class_means[,5], Sigma = class_covariances[,,5])
points6 <- mvrnorm(n = class_sizes[2] / 4, mu = class_means[,6], Sigma = class_covariances[,,6])
points7 <- mvrnorm(n = class_sizes[1] / 4, mu = class_means[,7], Sigma = class_covariances[,,7])
points8 <- mvrnorm(n = class_sizes[1] / 4, mu = class_means[,8], Sigma = class_covariances[,,8])
X <- rbind(points1, points3, points7, points8, points2, points4, points5, points6)
colnames(X) <- c("x1", "x2")

# generate corresponding labels
y <- c(rep(1, class_sizes[1]), rep(0, class_sizes[2]))

# write data to a file
write.csv(x = cbind(X, y), file = "lab05_data_set.csv", row.names = FALSE)

# plot data points generated
plot(X[y == 1, 1], X[y == 1, 2], type = "p", pch = 19, col = "red", las = 1,
     xlim = c(-6, +6), ylim = c(-6, +6),
     xlab = "x1", ylab = "x2")
points(X[y == 0, 1], X[y == 0, 2], type = "p", pch = 19, col = "blue")
