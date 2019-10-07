#reading x_data into memory
x_data_set <- read.csv('hw01_data_set_images.csv', header = FALSE)

#reading y_data into memory
y_data_set <- read.csv('hw01_data_set_labels.csv', header = FALSE)

#changing y_data_set class labels
require(plyr)
y_data_set$V1 <- mapvalues(y_data_set$V1,
                           from=c('A', 'B', 'C', 'D', 'E'),
                           to=c(1, 2, 3, 4, 5))

#train-test split of x_data_set and y_data_set
a_x_training_data_set <- x_data_set[1:25, ]
a_y_training_labels <- data.frame(y_data_set[1:25, ])
colnames(a_y_training_labels) <- 'label'
a_x_test_data_set <- x_data_set[26:39, ]
a_y_test_labels <- data.frame(y_data_set[26:39, ])
colnames(a_y_test_labels) <- 'label'

b_x_training_data_set <- x_data_set[40:64, ]
b_y_training_labels <- data.frame(y_data_set[40:64, ])
colnames(b_y_training_labels) <- 'label'
b_x_test_data_set <- x_data_set[65:78, ]
b_y_test_labels <- data.frame(y_data_set[65:78, ])
colnames(b_y_test_labels) <- 'label'

c_x_training_data_set <- x_data_set[79:103, ]
c_y_training_labels <- data.frame(y_data_set[79:103, ])
colnames(c_y_training_labels) <- 'label'
c_x_test_data_set <- x_data_set[104:117, ]
c_y_test_labels <- data.frame(y_data_set[104:117, ])
colnames(c_y_test_labels) <- 'label'

d_x_training_data_set <- x_data_set[118:142, ]
d_y_training_labels <- data.frame(y_data_set[118:142, ])
colnames(d_y_training_labels) <- 'label'
d_x_test_data_set <- x_data_set[143:156, ]
d_y_test_labels <- data.frame(y_data_set[143:156, ])
colnames(d_y_test_labels) <- 'label'

e_x_training_data_set <- x_data_set[157:181, ]
e_y_training_labels <- data.frame(y_data_set[157:181, ])
colnames(e_y_training_labels) <- 'label'
e_x_test_data_set <- x_data_set[182:195, ]
e_y_test_labels <- data.frame(y_data_set[182:195, ])
colnames(e_y_test_labels) <- 'label'

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

#removing unuseful variables from the environment
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

#prior probabilities
class_priors <- sapply(X = 1:5, FUN = function(c) {mean(y_train_truth == c)})

#pcd estimations
temp_y <- matrix(rep(y_train_truth, 320), ncol = 320)
y1 <- temp_y == 1
num1 <- x_train * y1
num1 <- matrix(colSums(num1), ncol = 320)
denum1 <- matrix(rep(sum(y_train_truth == 1), 320), ncol = 320)
p1d <- num1 / denum1

y2 <- temp_y == 2
num2 <- x_train * y2
num2 <- matrix(colSums(num2), ncol = 320)
denum2 <- matrix(rep(sum(y_train_truth == 2), 320), ncol = 320)
p2d <- num2 / denum2

y3 <- temp_y == 3
num3 <- x_train * y3
num3 <- matrix(colSums(num3), ncol = 320)
denum3 <- matrix(rep(sum(y_train_truth == 3), 320), ncol = 320)
p3d <- num3 / denum3

y4 <- temp_y == 4
num4 <- x_train * y4
num4 <- matrix(colSums(num4), ncol = 320)
denum4 <- matrix(rep(sum(y_train_truth == 4), 320), ncol = 320)
p4d <- num4 / denum4

y5 <- temp_y == 5
num5 <- x_train * y5
num5 <- matrix(colSums(num5), ncol = 320)
denum5 <- matrix(rep(sum(y_train_truth == 5), 320), ncol = 320)
p5d <- num5 / denum5

#printing pcd vectors
print(p1d)
print(p2d)
print(p3d)
print(p4d)
print(p5d)

#safelog function to handle the case where x = 0
safelog <- function(x) {
  return (log(x + 1e-100))
}

#labeling a row according to the max value in that row
get_label <- function(row){
  rm <- max(row)
  if(row[1] == rm){
    return(1)
  }
  if(row[2] == rm){
    return(2)
  }
  if(row[3] == rm){
    return(3)
  }
  if(row[4] == rm){
    return(4)
  }
  if(row[5] == rm){
    return(5)
  }
}

#predicting labels for each row in a matrix
predict_labels <- function(m){
  return(matrix(apply(m, 1, function(x) get_label(x)), nrow = length(m[,1])))
}

#training score values
c1e1 <- x_train %*% safelog(matrix(rep(p1d, 125), nrow = 320))
c1e1 <- matrix(c1e1[, 1], nrow = 125)
c1e2 <- (matrix(1, nrow = 125, ncol = 320) - x_train) %*% 
  safelog(matrix(rep((matrix(1, nrow = 1, ncol = 320) - p1d), 125), nrow = 320))
c1e2 <- matrix(c1e2[, 1], nrow = 125)
c1e3 <- matrix(rep(class_priors[1], 125), nrow = 125)
train_g1 <- c1e1 + c1e2 + c1e3

c2e1 <- x_train %*% safelog(matrix(rep(p2d, 125), nrow = 320))
c2e1 <- matrix(c2e1[, 1], nrow = 125)
c2e2 <- (matrix(1, nrow = 125, ncol = 320) - x_train) %*% 
  safelog(matrix(rep((matrix(1, nrow = 1, ncol = 320) - p2d), 125), nrow = 320))
c2e2 <- matrix(c2e2[, 1], nrow = 125)
c2e3 <- matrix(rep(class_priors[2], 125), nrow = 125)
train_g2 <- c2e1 + c2e2 + c2e3

c3e1 <- x_train %*% safelog(matrix(rep(p3d, 125), nrow = 320))
c3e1 <- matrix(c3e1[, 1], nrow = 125)
c3e2 <- (matrix(1, nrow = 125, ncol = 320) - x_train) %*% 
  safelog(matrix(rep((matrix(1, nrow = 1, ncol = 320) - p3d), 125), nrow = 320))
c3e2 <- matrix(c3e2[, 1], nrow = 125)
c3e3 <- matrix(rep(class_priors[3], 125), nrow = 125)
train_g3 <- c3e1 + c3e2 + c3e3

c4e1 <- x_train %*% safelog(matrix(rep(p4d, 125), nrow = 320))
c4e1 <- matrix(c4e1[, 1], nrow = 125)
c4e2 <- (matrix(1, nrow = 125, ncol = 320) - x_train) %*% 
  safelog(matrix(rep((matrix(1, nrow = 1, ncol = 320) - p4d), 125), nrow = 320))
c4e2 <- matrix(c4e2[, 1], nrow = 125)
c4e3 <- matrix(rep(class_priors[4], 125), nrow = 125)
train_g4 <- c4e1 + c4e2 + c4e3

c5e1 <- x_train %*% safelog(matrix(rep(p5d, 125), nrow = 320))
c5e1 <- matrix(c5e1[, 1], nrow = 125)
c5e2 <- (matrix(1, nrow = 125, ncol = 320) - x_train) %*% 
  safelog(matrix(rep((matrix(1, nrow = 1, ncol = 320) - p5d), 125), nrow = 320))
c5e2 <- matrix(c5e2[, 1], nrow = 125)
c5e3 <- matrix(rep(class_priors[5], 125), nrow = 125)
train_g5 <- c5e1 + c5e2 + c5e3

#training confusion matrix
training_scores <- cbind(train_g1, train_g2, train_g3, train_g4, train_g5)
y_train_hat <- predict_labels(training_scores)
train_confusion_matrix <- table(y_train_hat, y_train_truth)
print(train_confusion_matrix)

#test score values
c1e1 <- x_test %*% safelog(matrix(rep(p1d, 70), nrow = 320))
c1e1 <- matrix(c1e1[, 1], nrow = 70)
c1e2 <- (matrix(1, nrow = 70, ncol = 320) - x_test) %*% 
  safelog(matrix(rep((matrix(1, nrow = 1, ncol = 320) - p1d), 70), nrow = 320))
c1e2 <- matrix(c1e2[, 1], nrow = 70)
c1e3 <- matrix(rep(class_priors[1], 70), nrow = 70)
test_g1 <- c1e1 + c1e2 + c1e3

c2e1 <- x_test %*% safelog(matrix(rep(p2d, 70), nrow = 320))
c2e1 <- matrix(c2e1[, 1], nrow = 70)
c2e2 <- (matrix(1, nrow = 70, ncol = 320) - x_test) %*% 
  safelog(matrix(rep((matrix(1, nrow = 1, ncol = 320) - p2d), 70), nrow = 320))
c2e2 <- matrix(c2e2[, 1], nrow = 70)
c2e3 <- matrix(rep(class_priors[2], 70), nrow = 70)
test_g2 <- c2e1 + c2e2 + c2e3

c3e1 <- x_test %*% safelog(matrix(rep(p3d, 70), nrow = 320))
c3e1 <- matrix(c3e1[, 1], nrow = 70)
c3e2 <- (matrix(1, nrow = 70, ncol = 320) - x_test) %*% 
  safelog(matrix(rep((matrix(1, nrow = 1, ncol = 320) - p3d), 70), nrow = 320))
c3e2 <- matrix(c3e2[, 1], nrow = 70)
c3e3 <- matrix(rep(class_priors[3], 70), nrow = 70)
test_g3 <- c3e1 + c3e2 + c3e3

c4e1 <- x_test %*% safelog(matrix(rep(p4d, 70), nrow = 320))
c4e1 <- matrix(c4e1[, 1], nrow = 70)
c4e2 <- (matrix(1, nrow = 70, ncol = 320) - x_test) %*% 
  safelog(matrix(rep((matrix(1, nrow = 1, ncol = 320) - p4d), 70), nrow = 320))
c4e2 <- matrix(c4e2[, 1], nrow = 70)
c4e3 <- matrix(rep(class_priors[4], 70), nrow = 70)
test_g4 <- c4e1 + c4e2 + c4e3

c5e1 <- x_test %*% safelog(matrix(rep(p5d, 70), nrow = 320))
c5e1 <- matrix(c5e1[, 1], nrow = 70)
c5e2 <- (matrix(1, nrow = 70, ncol = 320) - x_test) %*% 
  safelog(matrix(rep((matrix(1, nrow = 1, ncol = 320) - p5d), 125), nrow = 320))
c5e2 <- matrix(c5e2[, 1], nrow = 70)
c5e3 <- matrix(rep(class_priors[5], 70), nrow = 70)
test_g5 <- c5e1 + c5e2 + c5e3

#test confusion matrix
test_scores <- cbind(test_g1, test_g2, test_g3, test_g4, test_g5)
y_test_hat <- predict_labels(test_scores)
test_confusion_matrix <- table(y_test_hat, y_test_truth)
print(test_confusion_matrix)
