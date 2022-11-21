#Upload of dataset and libraries

rm(list = ls())
ls()
set.seed(1234)
library(caret)
library(class)
library(gmodels)
library(pastecs)
library(ROSE)
library(tree)
library(randomForest)
library(ModelMetrics)
library(e1071)
library(MASS)
data=read.csv("C://Users//Luciana//Desktop//Machine learning//
heart_failure_clinical_records_dataset.csv", sep = ",", header = TRUE)
attach(data)
fileName <- "..//heart_failure_clinical_records_dataset.csv"
targetFeature <- "death_event"
cat("fileName: ", fileName, "\n", sep="")
cat("targetFeature: ", targetFeature, "\n", sep="")

#Overview and exploration phase

dim(data)
head(data, 10)
which(is.na(data)) 
str(data)
summary(data)
table(DEATH_EVENT)
100*round(prop.table(table(DEATH_EVENT)),3)
table(sex)
100*round(prop.table(table(sex)),3)
table(anaemia,DEATH_EVENT)
table(diabetes,DEATH_EVENT)
table(high_blood_pressure,DEATH_EVENT)
table(sex,DEATH_EVENT)
table(smoking,DEATH_EVENT)
par(mfrow=c(1,6))
boxplot(age~DEATH_EVENT)
boxplot(creatinine_phosphokinase~DEATH_EVENT)
boxplot(ejection_fraction~DEATH_EVENT)
boxplot(platelets~DEATH_EVENT)
boxplot(serum_creatinine~DEATH_EVENT)
boxplot(serum_sodium~DEATH_EVENT)
par(mfrow=c(1,1))
data_scaled=scale(data[,c(1,3,5,7,8,9,13)])
pairs(data_scaled[,-c(7)], col=ifelse(DEATH_EVENT==1,"red","green"), pch=20, 
cex=1.4)

#Pre-processing

data=data[,-c(12)]
data$anaemia=as.factor(data$anaemia)
data$diabetes=as.factor(data$diabetes)
data$high_blood_pressure=as.factor(data$high_blood_pressure)
data$sex=as.factor(data$sex)
data$smoking=as.factor(data$smoking)
data$DEATH_EVENT=as.factor(data$DEATH_EVENT)
data$creatinine_phosphokinase=log(data$creatinine_phosphokinase)
data$serum_creatinine=log(data$serum_creatinine)

#The analysis

matr_metrics=matrix(ncol=5, nrow=6)

confusion_matrix = function (actual, predicted){
  conf=table(actual,predicted)
  acc=sum(diag(conf))/sum(conf)
  recall=conf[1,1]/(conf[1,1]+conf[1,2])
  specificity=conf[2,2]/(conf[2,1]+conf[2,2])
  acc_balanced=(recall+specificity)/2
  curve=roc.curve(actual,predicted)
  auc_curve=curve$auc
  return(c(acc,acc_balanced,recall,specificity,auc_curve))
}

new_matr_1=matrix(ncol=5, nrow=100)
for(i in 1:100){
  trainIndex=createDataPartition(data$DEATH_EVENT, p=0.75, list=FALSE)
  data_train=data[trainIndex,]
  data_val=data[-trainIndex,]
  reg_model_new = glm(data_train[,12] ~ ., family=binomial(link="logit"),
   data=data_train[,-12])
  data_val_pred = predict(reg_model_new, data_val[,-12], type = "response")
  data_val_pred_bin = as.numeric(data_val_pred)
  data_val_pred_bin[data_val_pred_bin>=0.5]=1
  data_val_pred_bin[data_val_pred_bin<0.5]=0
  confusion_matrix(data_val[,12],data_val_pred_bin)
  new_matr_1[i,]=confusion_matrix(data_val[,12], data_val_pred_bin)
}
summary(lin_reg_model_new)
for(i in 1:ncol(matr_metrics)){
  matr_metrics[1,i]=mean(new_matr_1[,i])
}

new_matr_2=matrix(ncol=5, nrow=100)
rf_mod=randomForest(data_train[,-12], data_train[,12] , data=data_train, 
importance=TRUE, proximity=TRUE, ntree =1000)
plot(rf_mod)
t=tuneRF(data_train[,-12],data_train[,12], stepFactor = 0.5, plot=T,
 ntreeTry=300, trace = T, improve=0.05)
for(i in 1:100){
  trainIndex=createDataPartition(data$DEATH_EVENT, p=0.75, list=FALSE)
  data_train=data[trainIndex,]
  data_val=data[-trainIndex,]
  rf_mod= randomForest(data_train[,-12], data_train[,12] , data=data_train,
   importance=TRUE, proximity=TRUE, ntree =300)
  data_val_pred=predict(rf_mod, data_val[,-12], type = "response")
  new_matr_2[i,]=confusion_matrix(data_val[,12], data_val_pred)
}
for(i in 1:ncol(matr_metrics)){
  matr_metrics[2,i]=mean(new_matr_2[,i])
}

new_matr_3=matrix(ncol=5, nrow=100)
for(i in 1:100){
  trainIndex=createDataPartition(data$DEATH_EVENT, p=0.75, list=FALSE)
  data_train=data[trainIndex,]
  data_val=data[-trainIndex,]
  fit_tree=tree(DEATH_EVENT~., data=data_train, split="gini")
  fit_tree_pot=cv.tree(fit_tree, , prune.misclass, K=10)
  id.min=which.min(fit_tree$dev)
  tree_pruned_best=prune.misclass(fit_tree, best=id.min)
  plot(tree_pruned_best)
  data_val_pred=predict(tree_pruned_best, newdata=data_val, type="class")
  new_matr_3[i,]=confusion_matrix(data_val[,12], data_val_pred)
}
for(i in 1:ncol(matr_metrics)){
  matr_metrics[3,i]=mean(new_matr_3[,i])
}

new_matr_4=matrix(ncol=5, nrow=100)
for(i in 1:100){
  data_shuffled= data[sample(nrow(data)),] # shuffle the rows
  training_set_first_index= 1
  training_set_last_index= round(nrow(data_shuffled)*0.6)
  validation_set_first_index=training_set_last_index+1
  validation_set_last_index=round(nrow(data_shuffled)*0.9)
  test_set_first_index =validation_set_last_index+1
  test_set_last_index =nrow(data)
  data_train =data_shuffled[training_set_first_index:training_set_last_index,]
  data_validation = data_shuffled[validation_set_first_index:validation_set_last_index,]
  data_test =data_shuffled[test_set_first_index:test_set_last_index,]
  c_array = c(0.001, 0.01, 0.1, 1, 10)
  new=matrix(ncol=5, nrow=2)
  for(C in 1:5)
  {
    svm_mod = svm(data_train[,-c(2,4,6,10,11,12)], data_train[,12], 
    cost=c_array[C], data=data_train[,-12], method = "C-classification", 
    kernel = "linear")
    data_validation_PRED = predict(svm_mod, data_validation[,-c(2,4,6,10,11,12)])
    tab=table(data_validation_PRED, data_validation[,12])
    acc=sum(diag(tab))/nrow(as.matrix(data_validation_PRED))
    new[1,C]=c_array[C]
    new[2,C]=acc
  }
  param_c=new[1,which.max(new[2,])]
  svm_mod_new =svm(data_train[,-c(2,4,6,10,11,12)], data_train[,12], 
  cost=param_c, data=data_train[,-12], method = "C-classification", kernel = "linear")
  data_test_PRED =predict(svm_mod_new, data_test[,-c(2,4,6,10,11,12)])
  new_matr_4[i,]=confusion_matrix(data_test[,12],data_test_PRED)
}
for(i in 1:ncol(matr_metrics)){
  matr_metrics[4,i]=mean(new_matr_4[,i])
}

new_matr_5=matrix(ncol=5, nrow=100)
for(i in 1:100){
  data_shuffled = data[sample(nrow(data)),] # shuffle the rows
  training_set_first_index = 1
  training_set_last_index = round(nrow(data_shuffled)*0.6)
  validation_set_first_index=training_set_last_index+1
  validation_set_last_index=round(nrow(data_shuffled)*0.9)
  test_set_first_index = validation_set_last_index+1
  test_set_last_index = nrow(data)
  data_train =data_shuffled[training_set_first_index:training_set_last_index,]
  data_validation= data_shuffled[validation_set_first_index:validation_set_last_index,]
  data_test =data_shuffled[test_set_first_index:test_set_last_index,]
  c_array = c(0.001, 0.01, 0.1, 1, 10)
  new=matrix(ncol=5, nrow=2)
  for(C in 1:5)
  {
    svm_mod = svm(data_train[,-c(2,4,6,10,11,12)], data_train[,12], 
    cost=c_array[C], data=data_train[,-12], method = "C-classification", 
    kernel = "radial")
    data_validation_PRED= predict(svm_mod, data_validation[,-c(2,4,6,10,11,12)])
    tab=table(data_validation_PRED, data_validation[,12])
    acc=sum(diag(tab))/nrow(as.matrix(data_validation_PRED))
    new[1,C]=c_array[C]
    new[2,C]=acc
  }
  param_c=new[1,which.max(new[2,])]
  svm_mod_new =svm(data_train[,-c(2,4,6,10,11,12)], data_train[,12], 
  cost=param_c, data=data_train[,-12], method = "C-classification", kernel = "radial")
  data_test_PRED = predict(svm_mod_new, data_test[,-c(2,4,6,10,11,12)])
  new_matr_5[i,]=confusion_matrix(data_test[,12],data_test_PRED)
}
for(i in 1:ncol(matr_metrics)){
  matr_metrics[5,i]=mean(new_matr_5[,i])
}

new_matr_6=matrix(ncol=5, nrow=100)
for(i in 1:100){
  trainIndex=createDataPartition(data$DEATH_EVENT, p=0.75, list=FALSE)
  data_train=data[trainIndex,]
  data_val=data[-trainIndex,]
  ctrl =trainControl(method="repeatedcv",repeats = 3)
  knnFit= train(DEATH_EVENT ~ ., data = data_train, method = "knn",
   trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
  bestK=knnFit$bestTune
  data_val_pred =knn(train = data_train, test = data_val, cl = data_train[,12], 
  k=bestK)
  new_matr_6[i,]=confusion_matrix(data_val[,12],data_val_pred)
}
for(i in 1:ncol(matr_metrics)){
  matr_metrics[6,i]=mean(new_matr_6[,i])
}
colnames(matr_metrics)= c("accuracy", "balanced accuracy", "recall", 
"specificity", "ROC AUC")
rownames(matr_metrics)= c("logit model", "random forest", "decision tree",
 "linear svm", "gaussian svm", "knn")
matr_metrics

#The end

detach(data)
