x_Data <- read.csv('/Users/euludag14/Documents/MachineLearningCourse/engr421_dasc521_fall2019_hw01/hw01_images.csv', header = FALSE, row.names = NULL)
y_Data <- read.csv('/Users/euludag14/Documents/MachineLearningCourse/engr421_dasc521_fall2019_hw01/hw01_labels.csv', header = FALSE)


# I did the train and test split of x,y datasets
gender_x_training_Data <- x_Data[1:200, ]
gender_y_training_Label <- data.frame(y_Data[1:200, ])
colnames(gender_y_training_Label) <- 'label'

gender_x_test_Data <- x_Data[201:400, ]
gender_y_test_Label <- data.frame(y_Data[201:400, ])
colnames(gender_y_test_Label) <- 'label'

x_train <- data.matrix(gender_x_training_Data)
x_test <- data.matrix(gender_x_test_Data)

y_train <- data.matrix(gender_y_training_Label)
y_test <- data.matrix(gender_y_test_Label)


colMeans(x_train, na.rm = FALSE, dims = 1)

priorProbabilities <- sapply(X = 1:2, FUN = function(c) {mean(y_train == c)})
print(priorProbabilities)


