#reading data into memory
data <- read.csv(file = 'hw04_data_set.csv', header = TRUE)

#splitting data into train and test sets
train_data <- data[1:100, ]
test_data <- data[101:133, ]

############################################################
#regressogram function with data, bin_width, origin and x_max parameters
regressogram <- function(data, bin_width, origin, x_max){
  bin_number <- ceiling((data$x - origin) / bin_width)
  regressogram_data <- cbind(data, bin_number)
  bin_means <- aggregate(x = regressogram_data$y,
                         by = list(regressogram_data$bin_number),
                         FUN = mean)
  colnames(bin_means) <- c('bin_number', 'bin_mean')
  n <- (x_max - origin) / bin_width
  for(i in 1:n){
    if(!any(bin_means$bin_number == i)){
      bin_means <- rbind(bin_means, c(i, 0.0))
    }
  }
  return (bin_means[with(bin_means, order(bin_number)), ])
}

#regressogram parameters
regressogram_bin_width <- 3
regressogram_origin <- 0
regressogram_x_max <- 60

#regressogram of train_data with bin width: 3, origin: 0 and x_max: 60
train_data_regressogram <- regressogram(data = train_data,
                                        bin_width = regressogram_bin_width,
                                        origin = regressogram_origin,
                                        x_max = regressogram_x_max)

#plotting train_data, test_data and regressogram of train_data
plot(x = train_data$x,
     y = train_data$y,
     xlab = 'x',
     ylab = 'y',
     type = 'p',
     pch = 20,
     col = 'blue',
     xlim = c(regressogram_origin, regressogram_x_max))
points(x = test_data$x,
       y = test_data$y,
       type = 'p',
       pch = 20,
       col = 'red')
mock_data <- data.frame(1:regressogram_x_max)
mock_data <- cbind(mock_data,
                   as.vector(t(
                     matrix(rep(train_data_regressogram$bin_mean,
                                regressogram_bin_width),
                            nrow = (regressogram_x_max / regressogram_bin_width)))))
colnames(mock_data) <- c('x', 'bin_mean')
lines(x = mock_data$x,
      y = mock_data$bin_mean,
      xlab = 'x',
      ylab = 'y',
      type = 'S')
title(main = 'Regressogram with h = 3')
legend(x = max(data$x) + 4.8, y= max(data$y) + 8, legend = c('training', 'test'),
       col = c('blue', 'red'), pch = 20, xjust = 1, yjust = 1)

#calculating RMSE of regressogram for test_data
bin_number <- ceiling((test_data$x - regressogram_origin) / regressogram_bin_width)
test_data_regressogram <- cbind(test_data, bin_number)
test_data_regressogram <- merge(x = test_data_regressogram,
                                y = train_data_regressogram,
                                by = 'bin_number')
test_data_regressogram_rmse <- sqrt(
  sum((test_data_regressogram$y - test_data_regressogram$bin_mean)**2) /
    nrow(test_data_regressogram))
cat('Regressogram => RMSE is', test_data_regressogram_rmse,
    'when h is', regressogram_bin_width)

############################################################
#w function for running mean smoother
w <- function(x){
  x <- data.frame(abs(x))
  x <- apply(X = x, 
             MARGIN = c(1, 2), 
             FUN = function(e){
               if (e <= 0.5){
                 e <- 1
               }else{
                 e <- 0
               }
            })
  return (x)
}

#running mean smoother function with data_interval, data and bin_width parameters
rms <- function(data_interval, data, bin_width){
  g_head <- sapply(X = data_interval, 
                   FUN = function(x){
                     sum(w((x - data$x) / bin_width) * data$y) /
                       sum(w((x - data$x) / bin_width))
                   })
  result <- data.frame(data_interval)
  result <- cbind(result, g_head)
  colnames(result) <- c('x', 'rms')
  return (result)
}

#running mean smoother parameters
rms_bin_width <- 3
rms_origin <- 0
rms_x_max <- 60
rms_data_interval <- seq(from = rms_origin, to = rms_x_max, by = 0.1)

#running mean smoother of train_data with bin width: 3
train_data_rms <- rms(data_interval = rms_data_interval,
                      data = train_data,
                      bin_width = rms_bin_width)

#plotting train_data, test_data and running mean smoother of train_data
plot(x = train_data$x,
     y = train_data$y,
     xlab = 'x',
     ylab = 'y',
     type = 'p',
     pch = 20,
     col = 'blue',
     xlim = c(rms_origin, rms_x_max))
points(x = test_data$x,
       y = test_data$y,
       type = 'p',
       pch = 20,
       col = 'red')
lines(x = train_data_rms$x,
      y = train_data_rms$rms,
      xlab = 'x',
      ylab = 'y',
      type = 'l')
title(main = 'Running mean smoother with h = 3')
legend(x = max(data$x) + 4.8, y= max(data$y) + 8, legend = c('training', 'test'),
       col = c('blue', 'red'), pch = 20, xjust = 1, yjust = 1)

#calculating RMSE of running mean smoother for test_data
test_data_rms <- train_data_rms[test_data$x * 
                                  ((length(rms_data_interval) - 1) / 
                                     (rms_x_max - rms_origin)) + 
                                  1, ]
test_data_rms_rmse <- sqrt(
  sum((test_data$y - test_data_rms$rms)**2) /
    nrow(test_data_rms))
cat('Running Mean Smoother => RMSE is', test_data_rms_rmse,
    'when h is', rms_bin_width)

############################################################
#k function for kernel smoother
k <- function(x){
  return ((1 / sqrt(2 * pi)) * exp(-(x ** 2) / 2))
}

#kernel smoother function with data_interval, data and bin_width parameters
ks <- function(data_interval, data, bin_width){
  g_head <- sapply(X = data_interval, 
                   FUN = function(x){
                     sum(k((x - data$x) / bin_width) * data$y) /
                       sum(k((x - data$x) / bin_width))
                   })
  result <- data.frame(data_interval)
  result <- cbind(result, g_head)
  colnames(result) <- c('x', 'ks')
  return (result)
}

#kernel smoother parameters
ks_bin_width <- 1
ks_origin <- 0
ks_x_max <- 60
ks_data_interval <- seq(from = ks_origin, to = ks_x_max, by = 0.1)

#kernel smoother of train_data with bin width: 1
train_data_ks <- ks(data_interval = ks_data_interval,
                    data = train_data,
                    bin_width = ks_bin_width)

#plotting train_data, test_data and kernel smoother of train_data
plot(x = train_data$x,
     y = train_data$y,
     xlab = 'x',
     ylab = 'y',
     type = 'p',
     pch = 20,
     col = 'blue',
     xlim = c(ks_origin, ks_x_max))
points(x = test_data$x,
       y = test_data$y,
       type = 'p',
       pch = 20,
       col = 'red')
lines(x = train_data_ks$x,
      y = train_data_ks$ks,
      xlab = 'x',
      ylab = 'y',
      type = 'l')
title(main = 'Kernel smoother with h = 1')
legend(x = max(data$x) + 4.8, y= max(data$y) + 8, legend = c('training', 'test'),
       col = c('blue', 'red'), pch = 20, xjust = 1, yjust = 1)

#calculating RMSE of kernel smoother for test_data
test_data_ks <- train_data_ks[test_data$x * 
                                ((length(ks_data_interval) - 1) / 
                                   (ks_x_max - ks_origin)) + 
                                1, ]
test_data_ks_rmse <- sqrt(
  sum((test_data$y - test_data_ks$ks)**2) /
    nrow(test_data_ks))
cat('Kernel Smoother => RMSE is', test_data_ks_rmse,
    'when h is', ks_bin_width)

