mb11test<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb11test.csv",header=FALSE)
mb11train<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb11train.csv",header=FALSE)
mb12test<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb12test.csv",header=FALSE)
mb12train<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb12train.csv",header=FALSE)
mb13test<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb13test.csv",header=FALSE)
mb13train<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb13train.csv",header=FALSE)
mb21test<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb21test.csv",header=FALSE)
mb21train<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb21train.csv",header=FALSE)
mb22test<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb22test.csv",header=FALSE)
mb22train<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb22train.csv",header=FALSE)
mb23test<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb23test.csv",header=FALSE)
mb23train<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb23train.csv",header=FALSE)
mb31test<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb31test.csv",header=FALSE)
mb31train<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb31train.csv",header=FALSE)
mb32test<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb32test.csv",header=FALSE)
mb32train<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb32train.csv",header=FALSE)
mb33test<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb33test.csv",header=FALSE)
mb33train<-read.csv("C:\\Users\\Ika Candrawengi\\Documents\\kuliah\\machine learning\\Final Project\\DataSimulasi500 train400 test100\\mb33train.csv",header=FALSE)

"============================Discriminant Analysis=============================="
library(MVN)
library(MASS)
library(biotools)
library(DT)
library(lattice)
library(ggplot2)
library(generics)
library(caret)
library(MLmetrics)
library(mlbench)
library(randomForest)
library(pROC)
library(kernlab)
library(e1071)
library(dplyr)
library(nnet)
library(neuralnet)
library(logistf)
library(readr)
##==================================================================================##
analyze <- function(training,testing) {
###--------------------------LDA----------------------------------------##
  training[,1]<-as.factor(training[,1])
  testing[,1]<-as.factor(testing[,1])
  levels(training[,1]) <- c("F", "S")
  levels(testing[,1]) <- c("F", "S")
  DA1<-lda(formula=V1~.,data=training)
  predict1<-predict(DA1,testing[,-1])
  predict2<-predict(DA1,training[,-1])
  lda1<-confusionMatrix(predict1$class,testing[,1])
  lda2<-confusionMatrix(predict2$class,training[,1])
  AUClda1<-auc(roc(testing[,1],predict1$posterior[,1]))
  AUClda2<-auc(roc(training[,1],predict2$posterior[,1]))
###-------------------------Reg-Log------------------------------------###
  model<-glm(formula= V1~.,family=binomial(link="logit"), data=training)
  pred <- as.factor(ifelse(model$fitted.values < 0.5, "F", "S"))
  logit2<-confusionMatrix(pred,training[,1])
  cats.prob = predict(model, testing[,-1], type="response")
  pred2<-as.factor(ifelse(cats.prob<0.5,"F","S"))
  logit1<-confusionMatrix(pred2,testing[,1])
  back<-step(model)
 # back_pred<-predict(back,training[,-1],type="response")
  #back1<-as.factor(ifelse(back_pred<0.5,0,1))
  back_pred<-as.factor(ifelse(back$fitted.values <0.5,"F","S"))
  back_pred2<-predict(back,testing[,-1],type="response")
  back2<-as.factor(ifelse(back_pred2<0.5,"F","S"))
  stepreg1<-confusionMatrix(back_pred,training[,1])
  stepreg2<-confusionMatrix(back2,testing[,1])
  auc_reglog_train<-auc(roc(training[,1],back$fitted.values))
  auc_reglog_test<-auc(roc(testing[,1],back_pred2))
###------------------------Random Forest--------------------------------###
  set.seed(998)
  fitControl <- trainControl(method = "cv",
                             number = 10,
                             repeats = 10,
                             classProbs = TRUE,
                             summaryFunction = twoClassSummary,
                             search = "random")
  set.seed(825)
  rf_fit <- train(V1 ~ ., data = training, 
                  method = "rf",
                  metric = "ROC",
                  tuneLength = 10,
                  trControl = fitControl)
  bestmtry<-rf_fit[["bestTune"]][["mtry"]]
  test_predict<-predict(rf_fit,testing[,-1])
  train_predict<-predict(rf_fit,training[,-1])
  test_rf<-confusionMatrix(test_predict,testing[,1])
  train_rf<-confusionMatrix(train_predict,training[,1])
  test_rf
  pred.rf.train.auc<-predict(rf_fit,training[,-1],type="prob")
  pred.rf.test.auc<-predict(rf_fit,testing[,-1],type="prob")
  aucrftrain<-auc(roc(training[,1],pred.rf.train.auc[,1]))
  aucrftest<-auc(roc(testing[,1],pred.rf.test.auc[,1]))
###-------------------------------SVM Radial------------------------------###
  set.seed(825)
  svm.tune <- train(x=training[,-1],
                    y=training[,1],
                    method = "svmRadial",   # Radial kernel
                    tuneLength = 5,                   # 5 values of the cost function,  # Center and scale data
                    metric="ROC",
                    trControl=fitControl)
  svm_test<-predict(svm.tune,testing[,-1])
  svm_train<-predict(svm.tune,training[,-1])
  test_svmrad<-confusionMatrix(svm_test,testing[,1])
  train_svmrad<-confusionMatrix(svm_train,training[,1])
  pred.svmrad.train<-predict(svm.tune,training[,-1],type="prob")
  pred.svmrad.test<-predict(svm.tune,testing[,-1],type="prob")
  auc_svmrad_train<-auc(roc(training[,1],pred.svmrad.train[,1]))
  auc_svmrad_test<-auc(roc(testing[,1],pred.svmrad.test[,1]))
  ###--------------------------SVM Linear---------------------------------##
  set.seed(825)
  svm.tune2 <- train(x=training[,-1],
                     y=training[,1],
                     method = "svmLinear2",   # Radial kernel
                     tuneLength = 5,                   # 5 values of the cost function,  # Center and scale data
                     metric="ROC",
                     trControl=fitControl)
  svmlin_test<-predict(svm.tune2,testing[,-1])
  svmlin_train<-predict(svm.tune2,testing[,-1])
  test_svmlin<-confusionMatrix(svmlin_test, testing[,1])
  train_svmlin<-confusionMatrix(svmlin_train,testing[,1])
  pred.svmlin.train<-predict(svm.tune2,training[,-1],type="prob")
  pred.svmlin.test<-predict(svm.tune2,testing[,-1],type="prob")
  aucsvmlintrain<-auc(roc(training[,1],pred.svmlin.train[,1]))
  aucsvmlintest<-auc(roc(testing[,1],pred.svmlin.test[,1]))
###--------------------------Neural Network-------------------------##
  set.seed(825)
  nn.tune <- train(x=training[,-1],
                   y=as.factor(training[,1]),
                   method = "nnet",   # Radial kernel
                   tuneLength = 10,                   # 5 values of the cost function,  # Center and scale data
                   metric="ROC",
                   trControl=fitControl)
  prednn<-predict(nn.tune,training[,-1])
  pred.aucnn.train<-predict(nn.tune,training[,-1],type="prob")
  pred.aucnn.test<-predict(nn.tune,testing[,-1],type="prob")
  auc.nn.train<-auc(roc(training[,1],pred.aucnn.train[,1]))
  auc.nn.test<-auc(roc(testing[,1],pred.aucnn.test[,1]))
  nn1<-confusionMatrix(prednn, training[,1])
  prednntest<-predict(nn.tune,testing[,-1])
  nn2<-confusionMatrix(prednntest, testing[,1])
  print("===============FINAL==========")
  print("===================Testing Data=============================")
  Model<-c("Discriminant Analysis","Logistic Regression","Random Forest","SVM Radial","SVM Linear","Neural Network")
  Akurasi<-c(lda1[["overall"]][["Accuracy"]],logit1[["overall"]][["Accuracy"]],test_rf[["overall"]][["Accuracy"]],test_svmrad[["overall"]][["Accuracy"]],test_svmlin[["overall"]][["Accuracy"]],nn2[["overall"]][["Accuracy"]])
  Presisi<-c(lda1[["byClass"]][["Precision"]],logit1[["byClass"]][["Precision"]],test_rf[["byClass"]][["Precision"]],test_svmrad[["byClass"]][["Precision"]],test_svmlin[["byClass"]][["Precision"]],nn2[["byClass"]][["Precision"]])
  Sensitiviti<-c(lda1[["byClass"]][["Sensitivity"]],logit1[["byClass"]][["Sensitivity"]],test_rf[["byClass"]][["Sensitivity"]],test_svmrad[["byClass"]][["Sensitivity"]],test_svmlin[["byClass"]][["Sensitivity"]],nn2[["byClass"]][["Sensitivity"]])
  Specificiti<-c(lda1[["byClass"]][["Specificity"]],logit1[["byClass"]][["Specificity"]],test_rf[["byClass"]][["Specificity"]],test_svmrad[["byClass"]][["Specificity"]],test_svmlin[["byClass"]][["Specificity"]],nn2[["byClass"]][["Specificity"]])
  F1score<-c(lda1[["byClass"]][["F1"]],logit1[["byClass"]][["F1"]],test_rf[["byClass"]][["F1"]],test_svmrad[["byClass"]][["F1"]],test_svmlin[["byClass"]][["F1"]],nn2[["byClass"]][["F1"]])
  AUC<-c(AUClda2,auc_reglog_test,aucrftest,auc_svmrad_test,aucsvmlintest,auc.nn.test)
  data<-data.frame(Model,Akurasi,Presisi, Sensitiviti, Specificiti,F1score,AUC)
  write_csv(data,"testing.csv",append=TRUE, col_names = TRUE)
  print("===================Training Data=============================")
  Model2<-c("Discriminant Analysis","Logistic Regression","Random Forest","SVM Radial","SVM Linear","Neural Network")
  Akurasi2<-c(lda1[["overall"]][["Accuracy"]],logit1[["overall"]][["Accuracy"]],train_rf[["overall"]][["Accuracy"]],train_svmrad[["overall"]][["Accuracy"]],train_svmlin[["overall"]][["Accuracy"]],nn2[["overall"]][["Accuracy"]])
  Presisi2<-c(lda1[["byClass"]][["Precision"]],logit1[["byClass"]][["Precision"]],train_rf[["byClass"]][["Precision"]],train_svmrad[["byClass"]][["Precision"]],train_svmlin[["byClass"]][["Precision"]],nn2[["byClass"]][["Precision"]])
  Sensitiviti2<-c(lda1[["byClass"]][["Sensitivity"]],logit1[["byClass"]][["Sensitivity"]],train_rf[["byClass"]][["Sensitivity"]],train_svmrad[["byClass"]][["Sensitivity"]],train_svmlin[["byClass"]][["Sensitivity"]],nn2[["byClass"]][["Sensitivity"]])
  Specificiti2<-c(lda1[["byClass"]][["Specificity"]],logit1[["byClass"]][["Specificity"]],train_rf[["byClass"]][["Specificity"]],train_svmrad[["byClass"]][["Specificity"]],train_svmlin[["byClass"]][["Specificity"]],nn2[["byClass"]][["Specificity"]])
  F1score2<-c(lda1[["byClass"]][["F1"]],logit1[["byClass"]][["F1"]],train_rf[["byClass"]][["F1"]],train_svmrad[["byClass"]][["F1"]],train_svmlin[["byClass"]][["F1"]],nn2[["byClass"]][["F1"]])
  AUC2<-c(AUClda1,auc_reglog_train,aucrftrain,auc_svmrad_train,aucsvmlintrain,auc.nn.train)
  data2<-data.frame(Model2,Akurasi2,Presisi2, Sensitiviti2, Specificiti2,F1score2,AUC2)
  write_csv(data2,"Training Data",append=TRUE, col_names = TRUE)
  ###-------------------------------Feature Selection--------------------##
  ##--------------------logistic Reg------------------------------###
  sigma<-svm.tune$bestTune$sigma
  C<-svm.tune$bestTune$C
  cost<-svm.tune2$bestTune$cost
  correlationMatrix <- cor(training[,-1])
  cormatriks<-data.frame(correlationMatrix)
  print(correlationMatrix)
  write_csv(cormatriks,"Correlation of Variable.csv",append = TRUE,col_names = TRUE)
  findcorr<-findCorrelation(correlationMatrix,cutoff=0.5)
  highly<-data.frame(findcorr)
  write_csv(highly,"Highly Correlate.csv",append=TRUE,col_names = TRUE)
  set.seed(123)
  rfProfile1 <- rfe(training[,-1], training[,1],
                    rfeControl = rfeControl(functions =rfFuncs,
                                            number = 200), trControl=fitControl,method="rf")
  predictrf1<-predict(rfProfile1,training[,-1])
  predictrf2<-predict(rfProfile1,testing[,-1])
  pred.auc.fsrf.train<-predict(rfProfile1,training[,-1],type="prob")
  pred.auc.fsrf.test<-predict(rfProfile2,testing[,-1],type="prob")
  AUC_rffs_train<-auc(roc(training[,1]))
  AUC_rffs_test<-auc(roc(pred))
  fsrf1<-confusionMatrix(training[,1],predictrf1$pred)
  fsrf2<-confusionMatrix(testing[,1],predictrf2$pred)
  set.seed(123)
  svmProfile1 <- rfe(training[,-1], training[,1],
                     rfeControl = rfeControl(functions = caretFuncs,
                                             number = 200),
                     trControl = fitControl,
                     ## pass options to train()
                     method = "svmRadial")
  predsvmfsrad<-predict(svmProfile1,training[,-1])
  predsvmfsrad2<-predict(svmProfile1,testing[,-1])
  fssvmrd1<-confusionMatrix(training[,1],predsvmfsrad)
  fssvmrd2<-confusionMatrix(testing[,1],predsvmfsrad2)
  set.seed(123)
  svmProfile2 <- rfe(training[,-1], training[,1],
                     rfeControl = rfeControl(functions = caretFuncs,
                                             number = 200),
                     trControl = fitControl,
                     ## pass options to train()
                     method = "svmLinear2")
  predsvmlin<-predict(svmProfile2,training[,-1])
  predsvmlin2<-predict(svmProfile2,testing[,-1])
  fssvmlin1<-confusionMatrix(training[,1],predsvmlin)
  fssvmlin2<-confusionMatrix(testing[,1],predsvmlin2)
  print("=======================Using Feature Selection=======================")
  featuremodel<-c("Logistic Regression","Random Forest","SVM Radial","SVM Linear")
  Akurasifs<-c(stepreg1[["overall"]][["Accuracy"]],fsrf1[["overall"]][["Accuracy"]],fssvmrd1[["overall"]][["Accuracy"]],fssvmlin1[["overall"]][["Accuracy"]])
  Presisifs<-c(stepreg1[["byClass"]][["Precision"]],fsrf1[["byClass"]][["Precision"]],fssvmrd1[["byClass"]][["Precision"]],fssvmlin1[["byClass"]][["Precision"]])
  Sensitivitifs<-c(stepreg1[["byClass"]][["Sensitivity"]],fsrf1[["byClass"]][["Sensitivity"]],fssvmrd1[["byClass"]][["Sensitivity"]],fssvmlin1[["byClass"]][["Sensitivity"]])
  Specificitifs<-c(stepreg1[["byClass"]][["Specificity"]],fsrf1[["byClass"]][["Specificity"]],fssvmrd1[["byClass"]][["Specificity"]],fssvmlin1[["byClass"]][["Specificity"]])
  F1scorefs<-c(stepreg1[["byClass"]][["F1"]],fsrf1[["byClass"]][["F1"]],fssvmrd1[["byClass"]][["F1"]],fssvmlin1[["byClass"]][["F1"]])
  fstraining<-data.frame(featuremodel,Akurasifs,Presisifs,Sensitivitifs,Specificitifs,F1scorefs)
  write_csv(fstraining,"FS training.csv", append=TRUE, col_names = TRUE)
  Akurasifs2<-c(stepreg2[["overall"]][["Accuracy"]],fsrf2[["overall"]][["Accuracy"]],fssvmrd2[["overall"]][["Accuracy"]],fssvmlin2[["overall"]][["Accuracy"]])
  Presisifs2<-c(stepreg2[["byClass"]][["Precision"]],fsrf2[["byClass"]][["Precision"]],fssvmrd2[["byClass"]][["Precision"]],fssvmlin2[["byClass"]][["Precision"]])
  Sensitivitifs2<-c(stepreg2[["byClass"]][["Sensitivity"]],fsrf2[["byClass"]][["Sensitivity"]],fssvmrd2[["byClass"]][["Sensitivity"]],fssvmlin2[["byClass"]][["Sensitivity"]])
  Specificitifs2<-c(stepreg2[["byClass"]][["Specificity"]],fsrf2[["byClass"]][["Specificity"]],fssvmrd2[["byClass"]][["Specificity"]],fssvmlin2[["byClass"]][["Specificity"]])
  F1scorefs2<-c(stepreg2[["byClass"]][["F1"]],fsrf2[["byClass"]][["F1"]],fssvmrd2[["byClass"]][["F1"]],fssvmlin2[["byClass"]][["F1"]])
  fstraining2<-data.frame(featuremodel,Akurasifs2,Presisifs2,Sensitivitifs2,Specificitifs2,F1scorefs2)
  write_csv(fstraining2,"FS testing.csv",append=TRUE,col_names = TRUE)
  }
analyze(mb11train,mb11test)
analyze(mb12train,mb12test)
analyze(mb13train,mb13test)
analyze(mb21train,mb22test)

analyze(mb22train,mb22test)
analyze(mb23train,mb23test)
analyze(mb31train,mb31test)
analyze(mb32train,mb32test)
analyze(mb33train,mb33test)

