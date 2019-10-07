library(tree)

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

# construct an emseble by training decision trees on feature subsets
set.seed(421)
ensemble_size <- 100
feature_subset_size <- 75
N_test <- nrow(X_test)

predicted_probabilities <- array(0, c(N_test, K, ensemble_size))
for (t in 1:ensemble_size) {
  print(sprintf("training decision tree #%d", t))
  selected_features <- sample(1:D, feature_subset_size)
  decision_tree <- tree(formula = 'y ~ .', data = as.data.frame(cbind(X_train[, selected_features], y = as.factor(y_train))))
  predicted_probabilities[,,t] <- predict(decision_tree, newdata = as.data.frame(X_test[, selected_features]))
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

# construct an ensemble by training decision trees on sample subsets
set.seed(421)
ensemble_size <- 100
sample_subset_size <- 50
N_test <- nrow(X_test)

predicted_probabilities <- array(0, c(N_test, K, ensemble_size))
for (t in 1:ensemble_size) {
  print(sprintf("training decision tree #%d", t))
  selected_samples <- as.vector(sapply(1:K, function(c) {sample(which(y_train == c), sample_subset_size / K)}))
  decision_tree <- tree(formula = 'y ~ .', data = as.data.frame(cbind(X_train[selected_samples,], y = as.factor(y_train[selected_samples]))))
  predicted_probabilities[,,t] <- predict(decision_tree, newdata = as.data.frame(X_test))
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

