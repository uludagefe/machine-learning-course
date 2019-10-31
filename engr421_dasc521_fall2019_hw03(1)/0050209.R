setwd('/Users/euludag14/Documents/MachineLearningCourse/engr421_dasc521_fall2019_hw03(1)/')

images_data <- read.csv("hw03_images.csv",header=FALSE)
labels_data <- read.csv("hw03_labels.csv",header=FALSE)

initial_V_data <- read.csv("initial_V.csv",header=FALSE)
initial_W_data <- read.csv("initial_W.csv",header=FALSE)
#number of nodes in the hidden layer
H <- 20
#number of features
D <- ncol(images_data)

#number of observations
R <- nrow(images_data)
y_truth <- labels_data$V1
#number of classes
N <-max(y_truth)

#creation of an one-hot-encoded y matrix
y_one_hot_encoded <- matrix(0, nrow = R, ncol = N)
y_one_hot_encoded[cbind(1:R, y_truth)] <- 1


y_truth_training_set <- y_truth[1:500]
y_truth_test_set <- y_truth[500:1000]

y_truth_training_set_oh_encoded <- y_one_hot_encoded[1:500]
y_truth_test_set_oh_encoded <- y_one_hot_encoded[500:1000]


y_truth_training <- data.matrix(y_truth_training_set)
y_truth_test_matrix <- data.matrix(y_truth_test_set)

x_training <- data.matrix(images_data[1:500,])
x_test <- data.matrix(images_data[500:1000,])


#gradient function of v

v_gradient <- function(y_truth_training_set,y_predicted,x_data){
  transpose_x <- t(x_data)
  prediction_error<-(y_truth_training_set - y_predicted)
  return( transpose_x %*% prediction_error )
}

#gradient function of w

#from book 11.7 figure 11.29 
#note that H is the number of nodes in the hidden layer
w_gradient <- function(X, Z, V, y_truth_training_set_oh_encoded, y_predicted){
  prediction_error<-(y_truth_training_set_oh_encoded - y_predicted)
  #we need to crop the nodes beginning from the second row
  biased_nodes <- V[2:(H + 1),]
  prediction_error %*% t(biased_nodes)
  x_data_with_hidden_node_amount <- Z[,1:H]
  return (t(X) %*%  (  (prediction_error %*% t(biased_nodes)) * x_data_with_hidden_node_amount * (1-x_data_with_hidden_node_amount)   )  )

}
#the sigmoid function
sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}
#the safelog function
safelog <- function(a){
  return ( log( a + 1e-100))
}

#the softmax function is from lab 4 but we need minor changes as that function had 3 paramters w and w0, this time we need 2 only
softmax <- function(score){
  
  scores <- score
  scores <- exp(scores - matrix(apply(scores, MARGIN = 1, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}

eta<- 0.0005
epsilon<- 1e-3
max_iteration <- 500


W <- data.matrix(initial_W_data)
V <- data.matrix(initial_V_data)

errors <- c()
iteration_count <- 1

#Using safelog is about being protected from x=0 case
calculate_error <- function(y_truth_training_set_oh_encoded,y_prediction){
  return (-sum(y_truth_training_set_oh_encoded * safelog(y_prediction)))
}
while(iteration_count <= max_iteration){
  
  matrix_product <- sigmoid(cbind(1,x_training) %*% W)
  y_training_predicted <- softmax(cbind(1,matrix_product)%*% V)
  #dim(x_training) 500 784 thus we must bind the output layer 1's as coloum
  #dim(W) 785 20
  
  prediction_error<- calculate_error(y_truth_training_set_oh_encoded,y_training_predicted)
  errors <- c(errors,prediction_error)
  
  prev_v <- V
  prev_w <- W
  V <- V+ eta * v_gradient(y_truth_training_set_oh_encoded,y_training_predicted, cbind(1,matrix_product))
  W <- W + eta * w_gradient(cbind(1,x_training), matrix_product, V, y_truth_training_set_oh_encoded,y_training_predicted)
  
  if(iteration_count != 1){
    if(  sqrt(sum((prev_w - W)^2) + sum((prev_v - V)^2)) < epsilon ){
      break
    }
  }
  iteration_count<-iteration_count+1
}

plot(1:iteration_count, errors,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

copy_columns <- function(x,desired_coloumn_number){
  temp<-rep(x,desired_coloumn_number)
  return (matrix(temp,ncol=desired_coloumn_number*ncol(x),nrow=nrow(x)))
}

y_training_predicted <- apply(y_training_predicted, MARGIN = 1, FUN = which.max)
confusion_matrix_training <- table(y_training_predicted, y_truth_training)
print(confusion_matrix_training)
y_test_pred <- softmax(cbind(1,sigmoid(cbind(1,x_test) %*% W)),V)

y_test_predicted <- apply(y_test_pred, MARGIN = 1, FUN = which.max)
confusion_matrix_test <- table(y_test_predicted,y_truth_test_set)
print(confusion_matrix_test)







