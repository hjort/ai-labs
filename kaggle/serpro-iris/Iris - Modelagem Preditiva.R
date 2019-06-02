################################################################################
# Importar as bibliotecas necessárias

library(caret)

# library(rpart)
# library(randomForest)
# library(gbm)
# library(MASS)
# library(kernlab)
# library(RSNNS)
# library(klaR)

################################################################################
# Carregar os dados de treino

train_data <- read.csv(file = "iris-train.csv", header = TRUE, sep = ",")
head(train_data, 5)

################################################################################
# Dividir os dados entre treino e teste

# 70% para treino, 30% para teste
index <- createDataPartition(train_data$Species, p=0.70, list=FALSE)

testset <- train_data[-index,]
trainset <- train_data[index,]

################################################################################
# Selecionar os dados de treino e teste

X_train = trainset[,2:5]
y_train = trainset$Species
X_test = testset[,2:5]
y_test = testset$Species

################################################################################
# Definir algoritmos a serem usados

set.seed(42)

# methods = c("nb")
methods = c("rpart", "rf", "gbm", "lda", "knn", "svmLinear", "nnet", "mlp", "nb")
models = list()

################################################################################
# Treinar modelos com diferentes algoritmos

for (method in methods) {
  print(paste("Treinando o algoritmo", toupper(method), "..."))
  
  # K-Fold cross-validation
  train_control <- trainControl(method="cv", number=10)
  
  # treinar o modelo preditivo
  model <- train(x = X_train,
                 y = y_train,
                 method = method,
                 trControl = train_control,
                 metric = "Accuracy")
  print(model)
  models[[method]] <- model
  
  # executar predição e gerar matriz de confusão
  y_pred <- predict(object = model, newdata = X_test)
  cm = confusionMatrix(y_pred, y_test)
  print("Matriz de Confusão:")
  print(cm)
}

# Exibir os resultados de cada algoritmo
results <- resamples(models)
summary(results)
dotplot(results)

################################################################################
# Carregar os dados de avaliação (teste)

test_data <- read.csv(file = "iris-test.csv", header = TRUE, sep = ",")
head(test_data, 5)

################################################################################
# Gerar arquivos de envio para todos os algoritmos selecionados

for (method in names(models)) {
  model <- models[method]
  print(paste("Gerando arquivo para o algoritmo", toupper(method), "..."))
  
  # Prever os resultados usando o modelo já treinado
  y_pred <- predict(object = model, newdata = test_data[,2:5])
  names(y_pred) <- c("Species")
  
  # Preparar o arquivo de envio
  submission <- data.frame("Id" = test_data$Id, y_pred)
  write.csv(submission,
            file = paste("iris-submission-r-", method, ".csv", sep = ""), 
            quote = FALSE, 
            row.names = FALSE)
}

################################################################################
