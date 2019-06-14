# https://rpubs.com/rpadebet/269829
# https://cran.r-project.org/web/packages/caret/
# http://topepo.github.io/caret/
# https://cran.r-project.org/web/packages/caret/vignettes/caret.html

################################################################################
# Installing the required packages

install.packages("caret")
install.packages("tidyr")

################################################################################
# Loading the data

# Read the file into the R environment
data <- read.csv(file = "iris-train.csv", header = TRUE, sep = ",")

class(data)

# View the data
View(data)

# View the top few rows of the data in R console
head(data, 10)

# Assigning meaningful column names
colnames(data)<-c("Id","Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")
head(data,5)

################################################################################
# Splitting the Data into Training and Testing Sets

# Load the Caret package which allows us to partition the data
library(caret)
# We use the dataset to create a partition (80% training 20% testing)
index <- createDataPartition(data$Species, p=0.70, list=FALSE)
# select 20% of the data for testing
testset <- data[-index,]
# select 80% of data to train the models
trainset <- data[index,]

################################################################################
# Explore the Data

# Summarizing the Data

# Dimensions of the data
dim(trainset)

# Structure of the data
str(trainset)

# Summary of the data
summary(trainset)

# Levels of the prediction column
levels(trainset$Species)

dim(testset)
str(testset)
summary(testset)
levels(testset$Species)

################################################################################
# Visualization and Exploration

# Base R plots

## Histogram
hist(trainset$Sepal.Width)

## Box plot to understand how the distribution varies by class of flower
par(mfrow=c(1,4))
for(i in 2:5) {
  boxplot(trainset[,i], main=names(trainset)[i])
}

################################################################################
# Grammer of Graphics GGPLOTs

# begin by loading the library
library(ggplot2)

# Scatter plot
g <- ggplot(data=trainset, aes(x = Petal.Length, y = Petal.Width))
print(g)

g <-g + 
  geom_point(aes(color=Species, shape=Species)) +
  xlab("Petal Length") +
  ylab("Petal Width") +
  ggtitle("Petal Length-Width")+
  geom_smooth(method="lm")
print(g)

## Box Plot
box <- ggplot(data=trainset, aes(x=Species, y=Sepal.Length)) +
  geom_boxplot(aes(fill=Species)) + 
  ylab("Sepal Length") +
  ggtitle("Iris Boxplot") +
  stat_summary(fun.y=mean, geom="point", shape=5, size=4) 
print(box)

library(ggthemes)
## Histogram
histogram <- ggplot(data=iris, aes(x=Sepal.Width)) +
  geom_histogram(binwidth=0.2, color="black", aes(fill=Species)) + 
  xlab("Sepal Width") +  
  ylab("Frequency") + 
  ggtitle("Histogram of Sepal Width")+
  theme_economist()
print(histogram)

## Faceting: Producing multiple charts in one plot
library(ggthemes)
facet <- ggplot(data=trainset, aes(Sepal.Length, y=Sepal.Width, color=Species))+
  geom_point(aes(shape=Species), size=1.5) + 
  geom_smooth(method="lm") +
  xlab("Sepal Length") +
  ylab("Sepal Width") +
  ggtitle("Faceting") +
  theme_fivethirtyeight() +
  facet_grid(. ~ Species) # Along rows
print(facet)

################################################################################
# Getting Started with Machine Learning

#library(caret)
#install.packages("e1071")
#library(e1071)

# A) Decision Tree Classifiers

model.rpart <- train(x = trainset[,2:5],
                 y = trainset[,6],
                 method = "rpart",
                 metric = "Accuracy")

print(model.rpart)

plot(model.rpart$finalModel)
text(model.rpart$finalModel)

## Predictions on train dataset
pred <- table(predict(object = model.rpart$finalModel,
                    newdata = trainset[,2:5],
                    type="class"))
pred

## Checking the accuracy using a confusion matrix by comparing predictions to actual classifications
confusionMatrix(predict(object = model.rpart$finalModel,
                        newdata = trainset[,2:5],
                        type="class"),
                trainset$Species)

## Checking accuracy on the testdata set we created initially
pred_test <- predict(object = model.rpart$finalModel,
                   newdata = testset[,2:5],
                   type="class")
confusionMatrix(pred_test, testset$Species)

################################################################################

#install.packages("randomForest")
library(randomForest)

# B) Random Forest Algorithm

model.rf <- train(x = trainset[,2:5],
                     y = trainset[,6],
                     method = "rf",
                     metric = "Accuracy")
print(model.rf)

pred <- table(predict(object = model.rpart$finalModel,
                      newdata = trainset[,2:5],
                      type="class"))

confusionMatrix(pred, trainset$Species)

## Performance on the test set
pred_test<-predict(object = model.rf$finalModel,
                   newdata = testset[,1:4],
                   type="class")
confusionMatrix(pred_test, testset$Species)

################################################################################

# C) Gradient Boosting Method

model.gbm <- train(x = trainset[,2:5],
                  y = trainset[,6],
                  method = "gbm",
                  metric = "Accuracy")
print(model.gbm)

## Verify the accuracy on the training set
pred <- predict(object = model.gbm, newdata = trainset[,2:5])
confusionMatrix(pred, trainset$Species)

confusionMatrix(pred_test, testset$Species)

################################################################################

# D) K Means Clustering Model

# Since Kmeans is a random start algo, we need to set the seed to ensure reproduceability
set.seed(20)

irisCluster <- kmeans(iris[, 2:5], centers = 3, nstart = 20)
irisCluster

# Check the classification accuracy
table(irisCluster$cluster, iris$Species)

plot(iris[c("Sepal.Length", "Sepal.Width")], col=irisCluster$cluster)
points(irisCluster$centers[,c("Sepal.Length", "Sepal.Width")], col=1:3, pch=8, cex=2)

################################################################################

# E) Linear Discriminant Analysis

#library(caret)
#install.packages("MASS")
<<<<<<< HEAD
install.packages("e1071")

library(MASS)
set.seed(1000)

# Fit the model
model.lda<-train(x = trainset[,2:5],
=======
#library(MASS)

set.seed(1000)

# Fit the model
model.lda <- train(x = trainset[,2:5],
>>>>>>> 174537a13e78d1511afd0fb1c7aef054e6f20d83
                 y = trainset[,6],
                 method = "lda",
                 metric = "Accuracy")

# Print the model
print(model.lda)

## Verify the accuracy on the training set
<<<<<<< HEAD
pred<-predict(object = model.lda, newdata = testset[,2:5])
confusionMatrix(pred, testset$Species)
=======
pred <- predict(object = model.lda, newdata = trainset[,2:5])
confusionMatrix(pred, trainset$Species)
>>>>>>> 174537a13e78d1511afd0fb1c7aef054e6f20d83

## Performance on the test set
pred_test <- predict(object = model.lda, newdata = testset[,2:5])
confusionMatrix(pred_test, testset$Species)


testdata <- read.csv(file = "iris-test.csv", header = TRUE, sep = ",")
colnames(testdata)<-c("Id","Sepal.Length","Sepal.Width","Petal.Length","Petal.Width")
head(testdata,5)

pred_test<-predict(object = model.lda, newdata = testdata[,2:5])

View(pred_test)
class(pred_test)

################################################################################
# Summarizing the Models

# summarize accuracy of models
results <- resamples(list(TREE=model.rpart, 
                          RandomForest=model.rf, 
                          GBM=model.gbm, 
                          LDA=model.lda))
summary(results)

# Plotting the results
dotplot(results)

################################################################################
################################################################################
