#reading x_data_set into memory
x_data_set <- read.csv('hw03_data_set_images.csv', header = FALSE)

#reading y_data_set into memory
y_data_set <- read.csv('hw03_data_set_labels.csv', header = FALSE)

#changing y_data_set to dummy form
y_data_set <- fastDummies::dummy_cols(y_data_set)
y_data_set <- y_data_set[, 2:6]
y_data_set <- data.matrix(y_data_set)

#train-test split of x_data_set and y_data_set
a_x_training_data_set <- x_data_set[1:25, ]
a_y_training_labels <- data.frame(y_data_set[1:25, ])
a_x_test_data_set <- x_data_set[26:39, ]
a_y_test_labels <- data.frame(y_data_set[26:39, ])

b_x_training_data_set <- x_data_set[40:64, ]
b_y_training_labels <- data.frame(y_data_set[40:64, ])
b_x_test_data_set <- x_data_set[65:78, ]
b_y_test_labels <- data.frame(y_data_set[65:78, ])

c_x_training_data_set <- x_data_set[79:103, ]
c_y_training_labels <- data.frame(y_data_set[79:103, ])
c_x_test_data_set <- x_data_set[104:117, ]
c_y_test_labels <- data.frame(y_data_set[104:117, ])

d_x_training_data_set <- x_data_set[118:142, ]
d_y_training_labels <- data.frame(y_data_set[118:142, ])
d_x_test_data_set <- x_data_set[143:156, ]
d_y_test_labels <- data.frame(y_data_set[143:156, ])

e_x_training_data_set <- x_data_set[157:181, ]
e_y_training_labels <- data.frame(y_data_set[157:181, ])
e_x_test_data_set <- x_data_set[182:195, ]
e_y_test_labels <- data.frame(y_data_set[182:195, ])

#merging x_training_data_set, x_test_data_set, y_training_labels and y_test_labels
x_training_data_set <- rbind(a_x_training_data_set,
                             b_x_training_data_set,
                             c_x_training_data_set,
                             d_x_training_data_set,
                             e_x_training_data_set)
rownames(x_training_data_set) <- NULL
x_train <- data.matrix(x_training_data_set)

x_test_data_set <- rbind(a_x_test_data_set,
                         b_x_test_data_set,
                         c_x_test_data_set,
                         d_x_test_data_set,
                         e_x_test_data_set)
rownames(x_test_data_set) <- NULL
x_test <- data.matrix(x_test_data_set)

y_training_labels <- rbind(a_y_training_labels,
                           b_y_training_labels,
                           c_y_training_labels,
                           d_y_training_labels,
                           e_y_training_labels)
rownames(y_training_labels) <- NULL
y_train_truth <- data.matrix(y_training_labels)

y_test_labels <- rbind(a_y_test_labels,
                       b_y_test_labels,
                       c_y_test_labels,
                       d_y_test_labels,
                       e_y_test_labels)
rownames(y_test_labels) <- NULL
y_test_truth <- data.matrix(y_test_labels)

#removing useless variables from the environment for memory efficiency
remove(a_x_training_data_set, a_x_test_data_set,
       a_y_training_labels, a_y_test_labels,
       b_x_training_data_set, b_x_test_data_set,
       b_y_training_labels, b_y_test_labels,
       c_x_training_data_set, c_x_test_data_set,
       c_y_training_labels, c_y_test_labels,
       d_x_training_data_set, d_x_test_data_set,
       d_y_training_labels, d_y_test_labels,
       e_x_training_data_set, e_x_test_data_set,
       e_y_training_labels, e_y_test_labels,
       x_training_data_set, x_test_data_set,
       y_training_labels, y_test_labels,
       x_data_set, y_data_set)

#safelog function to handle the case where x = 0
safelog <- function(x) {
  return (log(x + 1e-100))
}

#sigmoid function
sigmoid <- function(m) {
  return (1 / (1 + exp(-m)))
}

#softmax function
softmax <- function(m){
  n <- nrow(m)
  k <- ncol(m)
  m <- exp(m)
  m_row_sums <-rowSums(m)
  m_row_sums <- rep(m_row_sums, k)
  m_row_sums <- matrix(m_row_sums, nrow = n, ncol = k)
  return(m/m_row_sums)
}

#gradient functions
gradient_v <- function(z, y_truth, y_predicted){
  return (-t(z) %*% (y_truth - y_predicted))
}

gradient_w <- function(x, z, v, y_truth, y_predicted){
  h <- nrow(v)
  return ((-t(x) %*% (((y_truth - y_predicted) %*% t(v)) * z * (1 - z)))[, 2:h])
}

#error function
error <- function(y_truth, y_predicted){
  return (-sum(y_truth * safelog(y_predicted)))
}

#seting learning parameters
eta <- 0.005
epsilon <- 1e-3
H <- 20
max_iteration <- 200
set.seed(521)

# getting the number of samples, the dimension of features and the number of classes
N <- nrow(x_train)
D <- ncol(x_train)
K <- ncol(y_train_truth)

#random initialization of w and v
w <- matrix(runif(n = (D + 1) * H, min = -0.01, max = +0.01), nrow = (D + 1), ncol = H)
v <- matrix(runif(n = (H + 1) * K, min = -0.01, max = +0.01), nrow = (H + 1), ncol = K)

#learning w and v using backpropagation algorithm under batch learning scenario
#updating w and v using gradient descent algorithm
iteration <- 1
error_values <- c()
while(TRUE){
  z <- sigmoid(m = (cbind(1, x_train) %*% w))
  y_train_predicted <- softmax(m = (cbind(1, z) %*% v))
  
  e <- error(y_truth = y_train_truth, y_predicted = y_train_predicted)
  error_values <- c(error_values, e)
  
  v_old <- v
  w_old <- w
  
  v <- v - eta * gradient_v(z = cbind(1, z),
                            y_truth = y_train_truth, y_predicted = y_train_predicted)
  w <- w - eta * gradient_w(x = cbind(1, x_train), z = cbind(1, z), v = v,
                            y_truth = y_train_truth, y_predicted = y_train_predicted)
  
  if (iteration != 1){
    if (abs(error_values[iteration] - error_values[iteration - 1]) < epsilon |
        iteration >= max_iteration) {
      break
    }
  }
  
  iteration <- iteration + 1
}

#plotting error values vs. iterations plot
plot(1:iteration, error_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

#finding the predicted class label for a image
get_label <- function(row){
  if(row[1] == 1){
    return(1)
  }
  if(row[2] == 1){
    return(2)
  }
  if(row[3] == 1){
    return(3)
  }
  if(row[4] == 1){
    return(4)
  }
  if(row[5] == 1){
    return(5)
  }
}

#predict_labels function applying get_label function to each image
predict_labels <- function(y_predicted){
  return(matrix(apply(y_predicted, 1, function(x) get_label(x)),
                nrow = length(y_predicted[,1])))
}

#calculating and printing the train_confusion_matrix
library(qlcMatrix)
y_train_max_val <- matrix(rep(rowMax(y_train_predicted)@x, K), ncol = K)
y_train_predicted <- 1 * (y_train_predicted == y_train_max_val)
y_train_hat <- predict_labels(y_train_predicted)
y_train <- predict_labels(y_train_truth)
train_confusion_matrix <- table(y_train_hat, y_train)
print(train_confusion_matrix)

#testing the model
#calculating and printing the test_confusion_matrix
y_test_predicted <- softmax(m = (cbind(1, sigmoid(m = (cbind(1, x_test) %*% w))) %*% v))
y_test_max_val <- matrix(rep(rowMax(y_test_predicted)@x, K), ncol = K)
y_test_predicted <- 1 * (y_test_predicted == y_test_max_val)
y_test_hat <- predict_labels(y_test_predicted)
y_test <- predict_labels(y_test_truth)
test_confusion_matrix <- table(y_test_hat, y_test)
print(test_confusion_matrix)

