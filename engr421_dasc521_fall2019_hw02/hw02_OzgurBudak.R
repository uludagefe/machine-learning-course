setwd('/home/emre/Documents/CollegeFiles/COMP421/MachineLearningCourse/engr421_dasc521_fall2019_hw02')
images <- read.csv("hw02_images.csv",header=FALSE)
labels <- read.csv("hw02_labels.csv",header=FALSE)
initial_W <- read.csv("initial_W.csv",header=FALSE)
initial_w0 <- read.csv("initial_w0.csv",header=FALSE)

y_truth <- labels$V1
y_truth_training <- head(y_truth, n=500)
y_truth_test <- tail(y_truth, n=500)
K <- max(y_truth)
N <- length(y_truth)


Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, y_truth)] <- 1

softmax <- function(X, W, w0) {
  scores <- cbind(X, 1) %*% rbind(W, t(w0))
  scores <- exp(scores - matrix(apply(scores, MARGIN = 1, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}

gradient_W <- function(X, Y_truth, Y_predicted) {
  return (-sapply(X = 1:ncol(Y_truth), function(c) colSums(matrix(Y_truth[,c] - Y_predicted[,c], nrow = nrow(X), ncol = ncol(X), byrow = FALSE) * X)))
}

gradient_w0 <- function(Y_truth, Y_predicted) {
  return (-colSums(Y_truth - Y_predicted))
}

safelog <- function(n) {
  
  return  (log(n + 1e-200))
  
}

eta <- 0.00004
epsilon <- 1e-3
max_iteration <- 500


W <- data.matrix(initial_W)
w0 <- data.matrix(initial_w0)

training_set= head(images, n=500)
Y_truth_training <- head(Y_truth, n=500)
test_set= tail(images, n=500)

X <- data.matrix(training_set)


iteration <- 0
objective_values <- c()



while (iteration < max_iteration) {
  Y_predicted <- softmax(X, W, w0)
  
  objective_values <- c(objective_values, -sum(Y_truth_training * safelog(Y_predicted)))
  
  W_old <- W
  w0_old <- w0
  
  W <- W - eta * gradient_W(X, Y_truth_training, Y_predicted)
  w0 <- w0 - eta * gradient_w0(Y_truth_training, Y_predicted)
  change <- sqrt(sum((w0 - w0_old)^2) + sum((W - W_old)^2)) 
  if ( change < epsilon) {
    break
  }
  iteration <- iteration + 1
  print(change)
  
}
print(W)




plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")


y_predicted <- apply(Y_predicted, MARGIN = 1, FUN = which.max)
confusion_matrix <- table(y_predicted, y_truth_training)
print(confusion_matrix)


y_pred_test <- softmax(data.matrix(test_set), W, w0)
y_predicted_test <- apply(y_pred_test, MARGIN = 1, FUN = which.max)
confusion_matrix_test <- table(y_predicted_test , y_truth_test)
print(confusion_matrix_test)

