# install.packages("e1071")
# install.packages("kernlab")
library(e1071)
library(kernlab)
# read the yelp dataset
setwd("C:/Users/Arun/Documents/NCSU MS/BI/Capstone")
yelpData <- read.csv("processed_data.csv")

# Get information about the dataset
str(yelpData)

# separate the numeric data columns from the Factor(string) data columns

num = sapply(yelpData, is.numeric)  # num contains numeric column indices
fact = sapply(yelpData, is.factor)  # fact contains the factor column indices
yelp_numeric = yelpData[, num]      # the numeric data
yelp_factor = yelpData[, fact]      # the factor(textual) data

# plot the histogram of the star ratings

remove_cols <- c(1)
yelp_numeric <- yelp_numeric[-remove_cols]

hist(yelp_numeric$stars,breaks = 20, col = "green") # Most of the scores lie in the range[5.5, 7.5]


# remove rows that have missing values
yelp_numeric_new <- na.omit(yelp_numeric) 

# scale the data
dat <- data.frame(lapply(yelp_numeric_new, function(x) scale(x, center = FALSE, scale = max(x, na.rm = TRUE))))

#Retaining the stars in the given range .i.e 0-5
dat$stars = yelp_numeric_new$stars  

#split the data in to trainset and test set
dat = data.frame(dat)
index <- 1:nrow(dat)

testindex <- sample(index, trunc(length(index)*1/4))
testset <- dat[testindex,]
trainset <- dat[-testindex,]

############################################ LINEAR REGRESSION MODEL #############################################
lm1 = lm(stars ~., data = trainset)
summary(lm1)
# Apply the model on the testset and get the predictted values
prediction = predict(lm1, testset[,-2])

# Calculate the Mean Square Error Values
mse <- mean((testset$stars - prediction)^2)
print(mse)

####################################### SUPPORT VECTOR MACHINES USING KERNEL #####################################
svm1<-svm(stars ~., data = trainset)
summary(svm1)
# Apply the model on the testset and get the predicted values
prediction = predict(svm1, testset[,-2],type="response")
# Calculate the Mean Squared Error Values
mse <- mean((testset$stars - prediction)^2)
print(mse)

