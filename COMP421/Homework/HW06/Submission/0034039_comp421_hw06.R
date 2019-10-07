library(MASS)
library(mvtnorm)

#Setting seed to 521 for the same data with data in homework description
set.seed(521)

#Setting density means
density_means <- matrix(c(+2.5, +2.5,
                          -2.5, +2.5,
                          -2.5, -2.5,
                          +2.5, -2.5,
                          +0.0, +0.0), 2, 5)

#Setting density covariances
density_covariances <- array(c(+0.8, -0.6, -0.6, +0.8,
                               +0.8, +0.6, +0.6, +0.8,
                               +0.8, -0.6, -0.6, +0.8,
                               +0.8, +0.6, +0.6, +0.8,
                               +1.6, +0.0, +0.0, +1.6), c(2, 2, 5))

#Setting density sample sizes
density_sizes <- c(50, 50, 50, 50, 100)

#Sample size and data dimensions
N <- sum(density_sizes)
D <- dim(density_means)[1]

#Data generation with 5 Gaussian densities
points1 <- mvrnorm(n = density_sizes[1], 
                   mu = density_means[, 1], 
                   Sigma = density_covariances[, , 1])
points2 <- mvrnorm(n = density_sizes[2], 
                   mu = density_means[, 2], 
                   Sigma = density_covariances[, , 2])
points3 <- mvrnorm(n = density_sizes[3], 
                   mu = density_means[, 3], 
                   Sigma = density_covariances[, , 3])
points4 <- mvrnorm(n = density_sizes[4], 
                   mu = density_means[, 4], 
                   Sigma = density_covariances[, , 4])
points5 <- mvrnorm(n = density_sizes[5], 
                   mu = density_means[, 5], 
                   Sigma = density_covariances[, , 5])
data <- rbind(points1, points2, points3, points4, points5)
colnames(data) <- c("x1", "x2")

#Plotting data
plot(points1[, 1], points1[, 2], type = "p", pch = 20, col = "black", las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6), xlab = "x1", ylab = "x2")
points(points2[, 1], points2[, 2], type = "p", pch = 20, col = "black")
points(points3[, 1], points3[, 2], type = "p", pch = 20, col = "black")
points(points4[, 1], points4[, 2], type = "p", pch = 20, col = "black")
points(points5[, 1], points5[, 2], type = "p", pch = 20, col = "black")

#Setting k = 5
k <- 5

#Initialize centroid values randomly
centroids <- t(data[sample.int(n = N, size = k), ])

#k-means algorithm for 2 iterations
k_means_max_iteration <- 2
for(i in 1 : k_means_max_iteration){
  #E step
  b_k <- c()
  for(c in 1 : k){
    centroid <- centroids[, c]
    b_k <- cbind(b_k, 
                 matrix(data = apply(X = data, 
                                     MARGIN = 1, 
                                     FUN = function(x){
                                       sqrt(sum((x - centroid)**2))
                                       }), 
                        ncol = 1))
  }
  b_k <- matrix(data = apply(b_k, MARGIN = 1, FUN = function(x){which.min(x)}),
                ncol = 1)
  data_b_k <- cbind(data, b_k)
  
  #M step
  new_centroids <- aggregate(data_b_k, by = list(data_b_k[, 3]), FUN = mean)
  for(c in new_centroids[, 4]){
    centroids[, c] <- c(new_centroids[new_centroids[, 4] == c, 2], 
                        new_centroids[new_centroids[, 4] == c, 3])
  }
}

#Mean vector initialization for EM algorithm
mean_vectors <- centroids

#Estimating prior probabilities for EM algorithm
prior_probs <- c()
for(c in 1 : k){
  prior_probs <- c(prior_probs, length(data_b_k[data_b_k[, 3] == c, 1]) / N)
}

#Estimating covariance matrices for EM algorithm
covar_mat <- c()
for(c in 1 : k){
  covar_mat <- c(covar_mat, 
                 (apply(X = data_b_k[data_b_k[, 3] == c, 1:2], 
                        MARGIN = 1, 
                        FUN = function(x){x- mean_vectors[, c]}) %*% 
                    t(apply(X = data_b_k[data_b_k[, 3] == c, 1:2], 
                            MARGIN = 1, 
                            FUN = function(x){x- mean_vectors[, c]}))) / 
                   length(data_b_k[data_b_k[, 3] == c, 1]))
}
covar_mat <- array(covar_mat, c(D, D, k))

#EM algorithm for 100 iterations
EM_max_iteration <- 100
for(i in 1 : EM_max_iteration){
  #E step
  h_k <- c()
  for(c in 1 : k){
    h_k <- cbind(h_k, 
                 mvtnorm::dmvnorm(x = data, 
                                  mean = mean_vectors[, c], 
                                  sigma = covar_mat[, , c]) * prior_probs[c])
  }
  h_k <- h_k / matrix(data = rep(rowSums(x = h_k), k), ncol = k)
  
  #M step
  prior_probs <- c()
  covar_mat <- c()
  for(c in 1 : k){
    mean_vectors[, c] <- colSums(data * 
                                   matrix(data = rep(h_k[, c], D), ncol = D)) / 
      sum(h_k[, c])
    prior_probs <- c(prior_probs, sum(h_k[, c]) / N)
    covar_mat <- c(covar_mat, 
                   ((t(matrix(data = rep(x = h_k[, c], D), ncol = D)) * 
                       apply(X = data, 
                             MARGIN = 1, 
                             FUN = function(x){(x- mean_vectors[, c])})) %*% 
                      t(t(matrix(data = rep(x = h_k[, c], D), ncol = D)) * 
                          apply(X = data, 
                                MARGIN = 1, 
                                FUN = function(x){x- mean_vectors[, c]}))) / 
                     sum(h_k[, c]))
  }
  covar_mat <- array(covar_mat, c(D, D, k))
  
  #EM algorithm results
  labels <- matrix(data = apply(h_k, MARGIN = 1, FUN = function(x){which.max(x)}),
                   ncol = 1)
  data_h_k <- cbind(data, labels)
}

#Printing mean vectors
print(t(mean_vectors))

#Plotting results
plot(data_h_k[data_h_k[, 3] == 1, 1], data_h_k[data_h_k[, 3] == 1, 2], 
     type = "p", pch = 20, col = "orange", las = 1, 
     xlim = c(-6, 6), ylim = c(-6, 6), xlab = "x1", ylab = "x2")
points(data_h_k[data_h_k[, 3] == 2, 1], data_h_k[data_h_k[, 3] == 2, 2], 
       type = "p", pch = 20, col = "red")
points(data_h_k[data_h_k[, 3] == 3, 1], data_h_k[data_h_k[, 3] == 3, 2], 
       type = "p", pch = 20, col = "green")
points(data_h_k[data_h_k[, 3] == 4, 1], data_h_k[data_h_k[, 3] == 4, 2], 
       type = "p", pch = 20, col = "blue")
points(data_h_k[data_h_k[, 3] == 5, 1], data_h_k[data_h_k[, 3] == 5, 2], 
       type = "p", pch = 20, col = "purple")

mock_x <- seq(from = -6, to = 6, by = 0.1)
mock_y <- mock_x
for(c in 1 : k){
  real_density <- matrix(data = 0, nrow = length(mock_x), ncol =  length(mock_x))
  em_density <- matrix(data = 0, nrow = length(mock_x), ncol =  length(mock_x))
  for (i in 1 :  length(mock_x)) {
    for (j in 1 :  length(mock_y)) {
      real_density[i, j] <- mvtnorm::dmvnorm(c(mock_x[i], mock_y[j]), 
                                             mean = density_means[, c], 
                                             sigma = density_covariances[, , c])
      em_density[i, j] <- mvtnorm::dmvnorm(c(mock_x[i], mock_y[j]), 
                                           mean = mean_vectors[, c], 
                                           sigma = covar_mat[, , c])
    }
  }
  contour(x = mock_x, y = mock_x, z = real_density, levels = .05, 
          col="black", lty = 'dashed', drawlabels = FALSE, add = TRUE)
  contour(x = mock_x, y = mock_x, z = em_density, levels = .05, 
          col="black", drawlabels = FALSE, add = TRUE)
}

