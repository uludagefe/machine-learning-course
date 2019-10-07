library(AUC)
library(MASS)
library(xgboost)

train_x <- read.csv(file = 'training_data.csv', header = TRUE)
train_y <- as.factor(read.csv("training_labels.csv", header = FALSE)[,1])
test_x <- read.csv(file = 'test_data.csv', header = TRUE)

#Categorical variables:
#F03
#F05
#F06
#F63

#Continuous variables:
#F01
#F02
#F07
#F08
#F09
#F10
#F11
#F12
#F61
#F62

#Rest of the features are binary

cont_variables <- c('F01', 'F02', 'F07', 'F08', 'F09', 
                    'F10', 'F11', 'F12', 'F61', 'F62')
train_x_cont <- train_x[, (names(train_x) %in% cont_variables)]
train_x_rest <- train_x[, !(names(train_x) %in% cont_variables)]

train_pca <- prcomp(x = train_x_cont, center = TRUE, scale. = TRUE)
plot(x = 1:ncol(train_x_cont), 
     y = cumsum(train_pca$sdev**2 / sum(train_pca$sdev^2)), 
     xlab = 'PC', ylab = 'Cum. % of variance')
summary(train_pca)

pca_dim <- 6
train_x_cont_pca <- train_pca$x[, 1 : pca_dim] %*% 
  t(train_pca$rotation[, 1 : pca_dim])
train_x_cont_pca <- t((t(train_x_cont_pca) * train_pca$scale) + train_pca$center)
train_x <- cbind(train_x_cont_pca, train_x_rest)

grid_search_table <- expand.grid(eta = seq(from = 0.1, to = 1, by = 0.3), 
                                 max_depth = c(4, 8, 16, 32),
                                 gamma = c(0, 5, 10, 20), 
                                 nrounds = c(10, 20))
grid_search <- apply(X = grid_search_table, 
                     MARGIN = 1, 
                     FUN = function(row){
                       curr_eta <- row[['eta']]
                       curr_max_depth <- row[['max_depth']]
                       curr_gamma <- row[['gamma']]
                       curr_nrounds <- row[['nrounds']]
                       print(c('eta' = curr_eta, 
                               'max_depth' = curr_max_depth, 
                               'gamma' = curr_gamma, 
                               'nrounds' = curr_nrounds))
                       xgb_cv <- xgb.cv(data = as.matrix(train_x), 
                                        label = as.matrix(train_y), 
                                        nfold = 8, 
                                        stratified = TRUE, 
                                        showsd = TRUE, 
                                        early_stopping_rounds = 5, 
                                        metrics = 'auc', 
                                        'objective' = 'binary:logistic', 
                                        'eval_metric' = 'auc', 
                                        'booster' = 'gbtree', 
                                        'eta' = curr_eta, 
                                        'max_depth' = curr_max_depth, 
                                        'gamma' = curr_gamma, 
                                        nrounds = curr_nrounds)
                       xgb_cv_scores <- as.data.frame(xgb_cv$evaluation_log)
                       val_auc <- tail(xgb_cv_scores$test_auc_mean, 1)
                       train_auc <- tail(xgb_cv_scores$train_auc_mean, 1)
                       output <- return(c(curr_eta, curr_max_depth, 
                                          curr_gamma, curr_nrounds, 
                                          train_auc, val_auc))
                       })
grid_search_results <- as.data.frame(t(grid_search))
columns <- c('eta', 'max_depth', 'gamma', 
             'nrounds', 'train_auc', 'validation_auc')
names(grid_search_results) <- columns

xgb_model <- xgboost(booster = 'gbtree', eta = 0.4, 
                     max_depth = 8, gamma = 20, 
                     objective = 'binary:logistic', eval_metric = 'auc', 
                     data = as.matrix(train_x), label = as.matrix(train_y),  
                     verbose = 1, nrounds = 20)

xgb_predicted <- predict(xgb_model, as.matrix(train_x))
xgb_predicted_label <- as.numeric(xgb_predicted > 0.5)

confusion_matrix <- table(xgb_predicted_label, train_y)
print(confusion_matrix)
roc_curve <- roc(predictions = xgb_predicted, labels = train_y)
auc(roc_curve)
plot(roc_curve$fpr, roc_curve$tpr, lwd = 2, col = "blue", type = "b", las = 1)

test_x_cont <- test_x[, (names(test_x) %in% cont_variables)]
test_x_rest <- test_x[, !(names(test_x) %in% cont_variables)]

test_pca <- prcomp(x = test_x_cont, center = TRUE, scale. = TRUE)
plot(x = 1:ncol(test_x_cont), 
     y = cumsum(test_pca$sdev**2 / sum(test_pca$sdev^2)), 
     xlab = 'PC', ylab = 'Cum. % of variance')
summary(test_pca)

pca_dim <- 6
test_x_cont_pca <- test_pca$x[, 1 : pca_dim] %*% 
  t(test_pca$rotation[, 1 : pca_dim])
test_x_cont_pca <- t((t(test_x_cont_pca) * test_pca$scale) + test_pca$center)
test_x <- cbind(test_x_cont_pca, test_x_rest)

test_predicted <- predict(xgb_model, as.matrix(test_x))
test_predicted_label <- as.numeric(test_predicted > 0.5)
write.table(test_predicted, file = "0034039_comp421_hw07_test_predicted.csv", 
            row.names = FALSE, col.names = FALSE)

