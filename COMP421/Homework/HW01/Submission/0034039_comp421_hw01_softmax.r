#reading x_data into memory
x_data_set <- read.csv('hw01_data_set_images.csv', header = FALSE)

#reading y_data into memory
y_data_set <- read.csv('hw01_data_set_labels.csv', header = FALSE)

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

#removing unuseful variables from the environment for memory efficiency
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

#softmax function
softmax <- function(x, w, b){
  n <- length(x[,1])
  broad_b <- matrix(rep(b, n), ncol = 5, byrow = TRUE)
  m <- x %*% w + broad_b
  m <- exp(m)
  m_row_sums <-rowSums(m)
  m_row_sums <- rep(m_row_sums, 5)
  m_row_sums <- matrix(m_row_sums, nrow = n, ncol = 5)
  return(m/m_row_sums)
}

#gradient functions
gradient_w <- function(x_train, y_truth, y_predict){
   return(-t(t(y_truth - y_predict) %*% x_train))
}

gradient_b <- function(y_truth, y_predict){
  return(-colSums(y_truth - y_predict))
}

#seting learning parameters
eta <- 0.001
epsilon <- 0.001

#random initialization of weight and bias matrix
#where weight is a 320*5 matrix and bias is a 125*5 matrix
w <- matrix(runif(320*5, min = -0.01, max = 0.01), nrow = 320, ncol = 5)
b <- matrix(runif(5, min = -0.01, max = 0.01), nrow = 1, ncol = 5)

#learn w and b using gradient descent algorithm
iteration <- 1
error_values <- c()
while(TRUE) {
  y_train_predict <- softmax(x_train, w, b)
  
  error <- -sum(y_train_truth * log(y_train_predict))
  
  error_values <- c(error_values, error)
  
  w_old <- w
  b_old <- b
  
  w <- w - eta * gradient_w(x_train, y_train_truth, y_train_predict)
  b <- b - eta * gradient_b(y_train_truth, y_train_predict)
  
  if (sqrt(sum((b - b_old)^2) + sum((w - w_old)^2)) < epsilon) {
    break
  }
  
  iteration <- iteration + 1
}

plot(1:iteration, error_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

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

predict_labels <- function(y_predict){
  return(matrix(apply(y_predict, 1, function(x) get_label(x)), nrow = length(y_predict[,1])))
}

library(qlcMatrix)
y_train_max_val <- matrix(rep(rowMax(y_train_predict)@x, 5), ncol = 5)
y_train_predict <- 1 * (y_train_predict == y_train_max_val)
y_train_hat <- predict_labels(y_train_predict)
y_train <- predict_labels(y_train_truth)
train_confusion_matrix <- table(y_train_hat, y_train)
print(train_confusion_matrix)

#testing the model
y_test_predict <- softmax(x_test, w, b)
y_test_max_val <- matrix(rep(rowMax(y_test_predict)@x, 5), ncol = 5)
y_test_predict <- 1 * (y_test_predict == y_test_max_val)
y_test_hat <- predict_labels(y_test_predict)
y_test <- predict_labels(y_test_truth)
test_confusion_matrix <- table(y_test_hat, y_test)
print(test_confusion_matrix)

