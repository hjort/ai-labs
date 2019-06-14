################################################################################
# Loading the data

# Read the file into the R environment
data <- read.csv(file = "iris-train.csv", header = TRUE, sep = ",")

# View the data
#View(data)

# View the top few rows of the data in R console
#head(data, 10)

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

set.seed(1000)

# Fit the model
model.lda<-train(x = trainset[,2:5],
                 y = trainset[,6],
                 method = "lda",
                 metric = "Accuracy")

# Print the model
print(model.lda)

## Verify the accuracy on the training set
#pred<-predict(object = model.lda, newdata = testset[,2:5])
#confusionMatrix(pred, testset$Species)

################################################################################

## Performance on the test set
pred_test<-predict(object = model.lda, newdata = testset[,2:5])
confusionMatrix(pred_test, testset$Species)

################################################################################

testdata <- read.csv(file = "iris-test.csv", header = TRUE, sep = ",")
colnames(testdata)<-c("Id","Sepal.Length","Sepal.Width","Petal.Length","Petal.Width")
head(testdata,5)

pred_test <- predict(object = model.lda, newdata = testdata[,2:5])

#Id <- testdata$Id
#Species <- pred_test
#class(Id)
#class(Species)

submission <- data.frame(Id = testdata$Id, Species = pred_test)
#class(submission)
#submission

write.csv(submission, file = "iris-submission-r-lda.csv", quote = FALSE, row.names = FALSE)

################################################################################
