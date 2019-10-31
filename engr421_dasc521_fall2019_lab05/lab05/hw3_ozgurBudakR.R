images <- read.csv("hw03_images.csv",header=FALSE)
labels <- read.csv("hw03_labels.csv",header=FALSE)
initial_W <- read.csv("initial_W.csv",header=FALSE)
initial_V <- read.csv("initial_V.csv",header=FALSE)


y_truth <- labels$V1


H <- 20
K <- max(y_truth)
N <- length(y_truth)

Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, y_truth)] <- 1

y_truth_training<- data.matrix(head(labels, n=500))
y_truth_test<- data.matrix(tail(labels, n=500))

Y_truth_training <- head(Y_truth, n=500)
Y_truth_test <- tail(Y_truth, n=500)


training_set= head(images, n=500)
test_set= tail(images, n=500)

eta <- 0.0005
epsilon <- 1e-3
max_iteration <- 500
X= data.matrix(training_set)
iteration <- 0
objective_values <- c()



W= data.matrix(initial_W)
V= data.matrix(initial_V)



safelog <- function(n) {
  
  return  (log(n + 1e-200))
  
}


sigmoid <- function(n) {
  
  return  (1 / (1 + exp(-1 * n)))
  
}


softmax <- function(z, V) {
  scores <- z %*% V
  scores <- exp(scores - matrix(apply(scores, MARGIN = 1, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}







gradient_V <- function(Y_truth_training, Y_predicted, z_with_bias) {
  ans1 <- (Y_truth_training - Y_predicted)
  ans2 <- t(z_with_bias) %*% ans1
  
  return (ans2)
}


gradient_W <- function(Y_truth_training, Y_predicted, z, V, X_with_bias){
  ans1 <- ((Y_truth_training - Y_predicted) %*% t(V[2:(H + 1),]))
  ans2 <- (ans1 * z[,1:H] * (1 - z[,1:H]))
  ans3 <- t(X_with_bias) %*% ans2
  
  return( ans3)
}






while (iteration < 500) {
  X_with_bias= cbind(1,X)
  z= sigmoid(X_with_bias %*% W)
  z_with_bias = cbind(1,z)
  
  Y_predicted <- softmax(z_with_bias, V)
  
  objective_values <- c(objective_values, -sum(Y_truth_training * safelog(Y_predicted)))
  
  V_old <- V
  W_old <- W
  
  
  
  V <- V+ eta * gradient_V(Y_truth_training,Y_predicted, z_with_bias )
  
  W <- W + eta * gradient_W(Y_truth_training, Y_predicted, z, V, X_with_bias)
  
  change <- sqrt(sum((W - W)^2) + sum((V - V_old)^2)) 
  
  if(change < epsilon)
  {
    break
  }
  iteration <- iteration + 1
  print(change)
  
}
print(W)

print(V)

plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")


y_predicted <- apply(Y_predicted, MARGIN = 1, FUN = which.max)
confusion_matrix <- table(y_predicted, y_truth_training)
print(confusion_matrix)

X_test= data.matrix(test_set)

X_with_bias_test= cbind(1,X_test)
z_test= sigmoid(X_with_bias_test %*% W)
z_with_bias_test = cbind(1,z_test)

Y_predicted_test <- softmax(z_with_bias_test, V)



y_predicted_test <- apply(Y_predicted_test, MARGIN = 1, FUN = which.max)
confusion_matrix_test <- table(y_predicted_test, y_truth_test)
print(confusion_matrix_test)

