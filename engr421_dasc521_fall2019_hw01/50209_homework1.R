
x_Data <- read.csv('hw01_images.csv', header = FALSE, row.names = NULL)
y_Data <- read.csv('hw01_labels.csv', header = FALSE)

#plyr is a tool for a big data structure into homogenous pieces
require(plyr)
yData$col1 <- mapvalues(yData$col1,from=c('1','2'),to=c(1,2))

# I did the train and test split of x,y datasets
gender_x_training_Data <- x_Data[1:200, ]
gender_y_training_Label <- data.frame(y_Data[1:200, ])
colnames(gender_y_training_Label) <- 'label'

gender_x_test_Data <- x_Data[201:400, ]
gender_y_test_Label <- data.frame(y_Data[201:400, ])
colnames(gender_y_test_Label) <- 'label'


