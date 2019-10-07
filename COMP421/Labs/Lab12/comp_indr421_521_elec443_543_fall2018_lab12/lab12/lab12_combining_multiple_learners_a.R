# define Euclidean distance function
pdist <- function(X1, X2) {
  if (identical(X1, X2) == TRUE) {
    D <- as.matrix(dist(X1))
  }
  else {
    D <- as.matrix(dist(rbind(X1, X2)))
    D <- D[1:nrow(X1), (nrow(X1) + 1):(nrow(X1) + nrow(X2))]
  }
  return(D)
}

# read data into memory
training_digits <- read.csv("lab12_mnist_training_digits.csv", header = FALSE)
training_labels <- read.csv("lab12_mnist_training_labels.csv", header = FALSE)
test_digits <- read.csv("lab12_mnist_test_digits.csv", header = FALSE)
test_labels <- read.csv("lab12_mnist_test_labels.csv", header = FALSE)

# get X and y values
X_train <- training_digits / 255
y_train <- training_labels[,1]
X_test <- test_digits / 255
y_test <- test_labels[,1]

# get number of classes and number of features
K <- max(y_train)
D <- ncol(X_train)

# construct an ensemble by training knns on feature subsets
set.seed(421)
ensemble_size <- 100
feature_subset_size <- 75
N_test <- nrow(X_test)
k <- 5

predicted_probabilities <- array(0, c(N_test, K, ensemble_size))
for (t in 1:ensemble_size) {
  print(sprintf("training knn #%d", t))
  selected_features <- sample(1:D, feature_subset_size)
  distance_matrix <- pdist(X_test[, selected_features], X_train[, selected_features])
  for (c in 1:K) {
    predicted_probabilities[,c,t] <- sapply(1:N_test, function(row) {sum(y_train[order(distance_matrix[row,], decreasing = FALSE)[1:k]] == c) / k}) 
  }
}

single_accuracies <- rep(0, ensemble_size)
combined_accuracies <- rep(0, ensemble_size)
for (t in 1:ensemble_size) {
  y_predicted <- apply(predicted_probabilities[,,t], MARGIN = 1, FUN = which.max)
  single_accuracies[t] <- mean(y_predicted == y_test)
  prediction <- apply(predicted_probabilities[,,1:t,drop = FALSE], MARGIN = c(1, 2), FUN = mean)
  y_predicted <- apply(prediction, MARGIN = 1, FUN = which.max)
  combined_accuracies[t] <- mean(y_predicted == y_test)
}

plot(1:ensemble_size, combined_accuracies, type = "b", col = "blue", lwd = 2, las = 1, pch = 19,
     xlab = "Ensemble size", ylab = "Classification accuracy",
     ylim = c(min(single_accuracies, combined_accuracies), max(single_accuracies, combined_accuracies)))
points(1:ensemble_size, single_accuracies, pch = 19, col = "red")

# construct an ensemble by training knns on sample subsets
set.seed(421)
ensemble_size <- 100
sample_subset_size <- 50
N_test <- nrow(X_test)
k <- 5

predicted_probabilities <- array(0, c(N_test, K, ensemble_size))
for (t in 1:ensemble_size) {
  print(sprintf("training knn #%d", t))
  selected_samples <- as.vector(sapply(1:K, function(c) {sample(which(y_train == c), sample_subset_size / K)}))
  distance_matrix <- pdist(X_test, X_train[selected_samples,])
  for (c in 1:K) {
    predicted_probabilities[,c,t] <- sapply(1:N_test, function(row) {sum(y_train[selected_samples][order(distance_matrix[row,], decreasing = FALSE)[1:k]] == c) / k}) 
  }
}

single_accuracies <- rep(0, ensemble_size)
combined_accuracies <- rep(0, ensemble_size)
for (t in 1:ensemble_size) {
  y_predicted <- apply(predicted_probabilities[,,t], MARGIN = 1, FUN = which.max)
  single_accuracies[t] <- mean(y_predicted == y_test)
  prediction <- apply(predicted_probabilities[,,1:t,drop = FALSE], MARGIN = c(1, 2), FUN = mean)
  y_predicted <- apply(prediction, MARGIN = 1, FUN = which.max)
  combined_accuracies[t] <- mean(y_predicted == y_test)
}

plot(1:ensemble_size, combined_accuracies, type = "b", col = "blue", lwd = 2, las = 1, pch = 19,
     xlab = "Ensemble size", ylab = "Classification accuracy",
     ylim = c(min(single_accuracies, combined_accuracies), max(single_accuracies, combined_accuracies)))
points(1:ensemble_size, single_accuracies, pch = 19, col = "red")

