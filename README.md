# Data-Science--Machine-Learning-Project-Detect-Credit-Card-Fraud-in-R
 Here is my project code detils
 
 Blog Link:http://techllearners.blogspot.com/2020/07/data-science-project-detect-furad-with.html
 
 The aim of this R project is to build a classifier that can detect credit card fraudulent transactions. We will use a variety of machine learning algorithms
 that will be able to discern fraudulent from non-fraudulent one. By the end of this machine learning project, you will learn how to implement machine learning 
 algorithms to perform classification. 
 
 

library(ranger)
library(caret)
library(data.table)
library(lattice)00
library(ggplot2)

creditcard <- read.csv("C:/Users/Pranto/Desktop/dataset/Credit-Card-Dataset/creditcard.csv/creditcard.csv")




#data exploration

dim(creditcard)

head(creditcard)

tail(creditcard)

table(creditcard $Class)

summary(creditcard$Amount)

names(creditcard)

var(creditcard$Amount)

sd(creditcard$Amount)





#data manipulation

head(creditcard)

creditcard$Amount=scale(creditcard$Amount)
NewData=creditcard[,-c(1)]
head(NewData)






#data modeling

library(caTools)

set.seed(123)

data_sample = sample.split(NewData$Class,SplitRatio=0.80)

train_data = subset(NewData,data_sample==TRUE)

test_data = subset(NewData,data_sample==FALSE)

dim(train_data)

dim(test_data)



#Fitting Logistic Regression Model

Logistic_Model=glm(Class~.,test_data,family=binomial())

summary(Logistic_Model)

plot(Logistic_Model)



#Fitting a Decision Tree Model

library(rpart)
library(rpart.plot)

decisionTree_model <- rpart(Class ~ . , creditcard, method = 'class')

predicted_val <- predict(decisionTree_model, creditcard, type = 'class')

probability <- predict(decisionTree_model, creditcard, type = 'prob')

rpart.plot(decisionTree_model)



#Artificial Neural Network


library(neuralnet)

ANN_model =neuralnet (Class~.,train_data,linear.output=FALSE)
plot(ANN_model)

predANN=compute(ANN_model,test_data)
resultANN=predANN$net.result
resultANN=ifelse(resultANN>0.5,1,0)





#Gradient Boosting (GBM)

library(gbm, quietly=TRUE)


# Get the time to train the GBM model
system.time(
  model_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(train_data, test_data)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
  )
)



# Determine best iteration based on test data
gbm.iter = gbm.perf(model_gbm, method = "test")

model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)


#Plot the gbm model
plot(model_gbm)


# Plot and calculate AUC on test data
gbm_test = predict(model_gbm, newdata = test_data, n.trees = gbm.iter)
gbm_auc = roc(test_data$Class, gbm_test, plot = TRUE, col = "red")


print(gbm_auc)
