## Code for Artificial Neural Networks

# install.packages("neuralnet")
# install.packages("nnet")

library(neuralnet)
library(nnet)
library(e1071)
# read the yelp dataset
setwd("C:/Users/Arun/Documents/NCSU MS/BI/Capstone")
yelpData <- read.csv("processed_data.csv")

# Get information about the dataset

str(yelpData)
yelpData$title_year = as.factor(yelpData$title_year)

# separate the numeric data columns from the Factor(string) data columns

num = sapply(yelpData, is.numeric)  # num contains numeric column indices
fact = sapply(yelpData, is.factor)  # fact contains the factor column indices
yelp_numeric = yelpData[, num]      # the numeric data
yelp_factor = yelpData[, fact]      # the factor(textual) data

# plot the histogram of the star ratings
remove_cols <- c(1)
yelp_numeric <- yelp_numeric[-remove_cols]

hist(yelp_numeric$stars,breaks = 20, col = "green") # Most of the scores lie in the range[3.5, 4.5]


# remove rows that have missing values
yelp_numeric_new <- na.omit(yelp_numeric) 

# scale the data
dat <- data.frame(lapply(yelp_numeric_new, function(x) scale(x, center = FALSE, scale = max(x, na.rm = TRUE))))

dat$stars = yelp_numeric_new$stars/5

#split the data in to trainset and test set
dat = data.frame(dat)
index <- 1:nrow(dat)

testindex <- sample(index, trunc(length(index)*1/4))
testset <- dat[testindex,]
trainset <- dat[-testindex,]


# create the formula for neural networks
myform <- as.formula(paste0('stars ~ ',paste(names(trainset[!names(trainset) %in% 'stars']), collapse = ' + ')))

# train the neural network with 4 hidden layers using the trainset
net.sqrt <- neuralnet(myform, trainset, hidden = 3)

#plot the obtained neural network
plot(net.sqrt)

# Apply the Neural Network model on the testset
comp <- compute(net.sqrt, testset[,-2])

# Calculate the Mean Squared Error
mse <- mean((testset$stars - comp$net.result)^2)
print(mse)

