################################################################################
# Importar as bibliotecas necessárias

library(caret)

################################################################################
# Carregar os dados

data <- read.csv(file = "iris-train.csv", header = TRUE, sep = ",")

head(data, 5)

################################################################################
<<<<<<< HEAD
# Divir os dados entre treino e teste
=======
# Dividir os dados entre treino e teste
>>>>>>> 174537a13e78d1511afd0fb1c7aef054e6f20d83

# 70% para treino, 30% para teste
index <- createDataPartition(data$Species, p=0.70, list=FALSE)

testset <- data[-index,]
trainset <- data[index,]

################################################################################
# Treinar o modelo preditivo

set.seed(1000)

model.lda <- train(x = trainset[,2:5], 
<<<<<<< HEAD
                   y = trainset[,6],
                   method = "lda",
                   metric = "Accuracy")
=======
                    y = trainset[,6],
                    method = "lda",
                    metric = "Accuracy")
>>>>>>> 174537a13e78d1511afd0fb1c7aef054e6f20d83

print(model.lda)

################################################################################
# Verificar o desempenho do modelo

pred_test <- predict(object = model.lda, newdata = testset[,2:5])

confusionMatrix(pred_test, testset$Species)

################################################################################
# Carregar os dados de avaliação

testdata <- read.csv(file = "iris-test.csv", header = TRUE, sep = ",")

head(testdata, 5)

################################################################################
# Prever os resultados usando o modelo já treinado

pred_test <- predict(object = model.lda, newdata = testdata[,2:5])

################################################################################
# Preparar o arquivo de envio

submission <- data.frame(Id = testdata$Id, Species = pred_test)

head(submission, 5)

write.csv(submission, file = "iris-submission-r-lda.csv", quote = FALSE, row.names = FALSE)

################################################################################
