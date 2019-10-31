setwd('/Users/euludag14/Documents/MachineLearningCourse/engr421_dasc521_fall2019_hw03(1)/')

images_data <- read.csv("hw03_images.csv",header=FALSE)
labels_data <- read.csv("hw03_labels.csv",header=FALSE)

initial_V_data <- read.csv("initial_V.csv",header=FALSE)
initial_W_data <- read.csv("initial_W.csv",header=FALSE)

y_truth <- labels_data$V1
y_truth_training_set <- head(y_truth,n=500)
y_truth_test_set <- tail(y_truth,n=500)

y_truth_training_matrix <- data.matrix(y_truth_training_set)
y_truth_test_matrix <- data.matrix(y_truth_test_set)

x_training <- data.matrix(head(images_data,n=500))
x_test <- data.matrix(tail(images_data,n=500))

#number of samples
N <-max(y_truth)

#number of features
D <- ncol(images_data)

#gradient function of v
v_gradient <- function(y_truth,y_predicted,x_data){
  minus_transpose <- -t(x_data)
  bias<-(y_truth - rowSums(y_predicted)/ncol(y_predicted))
  return( minus_transpose %*% bias )
}
#gradient function of w

w_gradient <- function(y_truth, y_predicted,X, Z, V){
  number_of_obs <- nrow(V)
  bias<-(y_truth - rowSums(y_predicted)/ncol(y_predicted))
  print(dim(y_truth))
  print(dim(V))
  bias <- matrix(  rep(bias ,ncol(y_truth)),nrow=nrow(y_truth), ncol=ncol(y_truth)   )
  print(dim(bias))
  # transpose_V <- t(V)
  #  minus_transpose_X <- -t(X)
  #output_not_cropped <- (minus_transpose_X %*% ((bias %*% transpose_V) * 1-Z * Z))
  #output <- output_not_cropped[,2:number_of_obs]
  return 
}

#the sigmoid function
sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}
#the safelog function
safelog <- function(a){
  return ( log( a + 1e-100))
}

#the softmax function
softmax <- function(matrix){

  number_of_features <- ncol(matrix)
  matrix <- exp(matrix)
  number_of_obs <- nrow(matrix)
  matrix_row_sums <-rowSums(matrix)
  
  #I made a sequence of row sums below
  matrix_row_sums_sequence <- rep(matrix_row_sums, number_of_features)
  
  #Then I turned it into a matrix
  matrix_row_sums_matrix <- matrix(matrix_row_sums_sequence, nrow = number_of_obs, ncol = number_of_features)
  
  return(matrix/matrix_row_sums_matrix)
}

eta<- 0.0005
epsilon<- 1e-3
max_iteration <- 500
H <- 20

W <- data.matrix(initial_W_data)
V <- data.matrix(initial_V_data)

errors <- c()
iteration_count <- 1

calculate_error <- function(y_truth,y_prediction){
  return (-sum(t(y_truth) %*% safelog(y_prediction)))
}
#while(1){
  
  matrix_product <- sigmoid(cbind(1,x_training) %*% W)
  y_training_predicted <- softmax(cbind(1,matrix_product) %*% V)
  #dim(x_training) 500 784 thus we must bind the output layer 1's as coloum
  #dim(W) 785 20
  
  prediction_error<- calculate_error(y_truth_training_matrix,y_training_predicted)
  errors <- c(errors,prediction_error)
  
  delta_v <- eta * v_gradient(y_truth_training_matrix,y_training_predicted,cbind(1,matrix_product))
  temp_delta_v <- matrix(rep(delta_v,ncol(V)),nrow=nrow(V),ncol=ncol(V))
  V <- V - temp_delta_v
  
  delta_w <- eta * w_gradient(y_truth_training_matrix,y_training_predicted,cbind(1, x_training),matrix_product,V)
  W <- W - delta_w

  error_margin <- 1e+10
#  if(iteration_count != 1){
    error_margin<-abs(errors[iteration_count]-errors[iteration_count-1])
 # }
  if(error_margin < epsilon | iteration_count = max_iteration){
    break
  }
  iteration_count<-iteration_count+1
#}

plot(1:iteration_count, errors,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")








