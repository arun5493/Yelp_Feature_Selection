# install.packages("VGAM")
# install.packages("kernlab")
# install.packages("glmnet")
# install.packages("corrplot")
# install.packages("caret")
# install.packages("mlogit")
# install.packages("MASS")
# install.packages("Hmisc")
# install.packages("reshape2")
# install.packages("stats")

library(glmnet)
library(corrplot)
library(caret)
library(mlogit)
library(foreign)
library(MASS)
library(Hmisc)
library(reshape2)
library(stats)


dataset<- read.csv("processed_data.csv")
cols <- c(1:6,9)

sub_dataset <- dataset
sub_dataset <- sub_dataset
#sub_dataset$stars <- round(sub_dataset$stars)
#sub_dataset$stars <- as.factor(sub_dataset$stars)
sub_dataset <- sub_dataset[-cols]

########################## REMOVE RENDUNDANT ATTRIBUTES USING CORRELATION ####################
sub_dataset[is.na(sub_dataset)] <- 0
correlation <- cor(sub_dataset)
corrplot(correlation,order="hclust")
correlation <- as.data.frame(correlation)
pair_1 <- c()
pair_2 <- c()
val <- c()
for(i in 1:nrow(correlation))
{
  for(j in 1:ncol(correlation))
  {
    if((correlation[i,j] < -0.6 | correlation[i,j] > 0.6) && i != j )
    {
      pair_1 <- c(pair_1,i)
      pair_2 <- c(pair_2,j)
      val <- c(val,correlation[i,j])
    }
  }
}

# REMOVING REDUNDANT FEATURES 
remove_cols <- c(8,29)
sub_dataset <- sub_dataset[-remove_cols]

dat <- data.frame(lapply(sub_dataset, function(x) scale(x, center = FALSE, scale = max(x, na.rm = TRUE))))
dat$stars = sub_dataset$stars

sub_dataset = data.frame(dat)
sub_dataset$stars <- round(sub_dataset$stars)
sub_dataset$stars <- as.factor(sub_dataset$stars)


# DIVIDING DATA INTO TRAINING AND TEST SET 

set.seed(250)
col <- c(1,2,3 ,6 ,8 ,9 ,13 ,14, 16 ,17,20)
sub_dataset <- sub_dataset[col]
rownum <- sample(nrow(sub_dataset))
sub_dataset <- sub_dataset[rownum,]
train <- sub_dataset[1:5400,]
test <- sub_dataset[5401:7858,]


##################################### MODEL BUILDING #######################################

#################### ORDINAL REGRESSION ##############################
model <- polr(stars ~.,data = train, Hess=TRUE)
summary(model)
pred <- predict(model,test)

#Confusion matrix
t <- table(actual=test$stars,predicted=pred)
print(t)
print("The accuracy is:")
print(((t[1]+t[7]+t[13]+t[19]+t[25])/sum(t))*100)
# ACCURACY - 58% approximately


######################## MULTINOMIAL REGRESSION ##########################

library(VGAM)
mlog1 <- vglm(stars ~ ., data=train, family=multinomial())
summary(mlog1)
pred1 <- predict(mlog1,test[-2],type="response")
predictions <- apply(pred1, 1, which.max)
predictions[which(predictions=="1")] <- levels(test$stars)[1]
predictions[which(predictions=="2")] <- levels(test$stars)[2]
predictions[which(predictions=="3")] <- levels(test$stars)[3]
predictions[which(predictions=="4")] <- levels(test$stars)[4]
predictions[which(predictions=="5")] <- levels(test$stars)[5]

#Confusion matrix
t1 <- table(predictions, test$stars)    
print(t1)
# ACCURACY - 59% approximately



################# BINARY LOGISTIC REGRESSION - GLM #################################
glm_train <- train
glm_test <- test
glm_train$s1 <- glm_train$stars == "1"
glm_train$s2 <- glm_train$stars == "2"
glm_train$s3 <- glm_train$stars == "3"
glm_train$s4 <- glm_train$stars == "4"
glm_train$s5 <- glm_train$stars == "5"
glm_train$stars <- NULL

glm_test$s1 <- glm_test$stars == "1"
glm_test$s2 <- glm_test$stars == "2"
glm_test$s3 <- glm_test$stars == "3"
glm_test$s4 <- glm_test$stars == "4"
glm_test$s5 <- glm_test$stars == "5"
glm_test$stars <- NULL

fit.s1 <- glm(s1 ~ ., data=glm_train, family=binomial(link="logit"),control = list(maxit = 50))
fit.s2 <- glm(s2 ~ ., data=glm_train, family=binomial(link="logit"),control = list(maxit = 50))
fit.s3 <- glm(s3 ~ ., data=glm_train, family=binomial(link="logit"),control = list(maxit = 50))
fit.s4 <- glm(s4 ~ ., data=glm_train, family=binomial(link="logit"),control = list(maxit = 50))
fit.s5 <- glm(s5 ~ ., data=glm_train, family=binomial(link="logit"),control = list(maxit = 50))

print(summary(fit.s1))
print(summary(fit.s2))
print(summary(fit.s3))
print(summary(fit.s4))
print(summary(fit.s5))

log_pred1 <- predict(fit.s1,glm_test)
log_pred2 <- predict(fit.s2,glm_test)
log_pred3 <- predict(fit.s3,glm_test)
log_pred4 <- predict(fit.s4,glm_test)
log_pred5 <- predict(fit.s5,glm_test)

t1 <- table(predicted=log_pred1 > 0.5,actual=glm_test$s1)
t2 <- table(predicted=log_pred2 > 0.5,actual=glm_test$s2)
t3 <- table(predicted=log_pred3 > 0.5,actual=glm_test$s3)
t4 <- table(predicted=log_pred4 > 0.5,actual=glm_test$s4)
t5 <- table(predicted=log_pred5 > 0.5,actual=glm_test$s5)

print(t1)
print(t2)
print(t3)
print(t4)
print(t5)

# Running the binary logistic regression model for our dataset gives an accuracy of 100% but
# this is a case of overfitting due to the addition of many dummy variables (splitting the star 
# rating column which consists of 5 labels - 1,2,3,4,5 into 5 different binary columns one each
# for the star rating). Hence, we use multinomial logistic regression which better explains 
# the response variable for our dataset.