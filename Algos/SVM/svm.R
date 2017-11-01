"
SVM classification using various kernels and comparison of evaluation metrics
"


#intall required libraries
#install.packages(caret)
#install.packages(e1071)
#install.packages(R.matlab)
library(R.matlab)
library(caret)
library(e1071)

#Q6-A
#Read the Data
data=readMat("data file")

# X is input class Y.target is response class (two classes '1' and '2')
X = data$X
Y = data$Y.target

#Dataset Division Train=60%,Test=40%
set.seed(500)
trainIndex=sample(1:nrow(X),0.6*nrow(X),replace=F)
testIndex=setdiff(1:nrow(X),trainIndex)
Xtrain=X[trainIndex,]
Xtest=X[testIndex,]
Ytrain=Y[trainIndex]
Ytest=Y[testIndex]


#SVM for different kernels and Respective evaluation Metrics
# Tune hyper parameters according to the requirement, for instance precision vs recall estimate.
g = 1/ncol(Xtest)

linearMetrics = c()
svmLinear <- svm(Xtrain,Ytrain,type = 'C-classification',kernel='linear')
YpredLinear = predict(svmLinear,Xtest)
cmLinear<-confusionMatrix(Ytest,YpredLinear)
precisionLinear <- cmLinear$byClass['Pos Pred Value']    
recallLinear <- cmLinear$byClass['Sensitivity']
f_measureLinear <- 2 * ((precisionLinear * recallLinear) / (precisionLinear + recallLinear))
linearMetrics = c(precisionLinear,recallLinear,f_measureLinear)

svmPoly <- svm(Xtrain,Ytrain,type = 'C-classification',kernel='polynomial',gamma = g,coeff0 = 0.5)
YpredPoly = predict(svmPoly,Xtest)
cmPoly<-confusionMatrix(Ytest,YpredPoly)
precisionPoly <- cmPoly$byClass['Pos Pred Value']    
recallPoly <- cmPoly$byClass['Sensitivity']
f_measurePoly <- 2 * ((precisionPoly * recallPoly) / (precisionPoly + recallPoly))
polyMetrics = c(precisionPoly,recallPoly,f_measurePoly)

svmQuad <- svm(Xtrain,Ytrain,type = 'C-classification',kernel='polynomial',degree = 2,gamma = g,coeff0 = 0.1)
YpredQuad = predict(svmQuad,Xtest)
cmQuad<-confusionMatrix(Ytest,YpredQuad)
precisionQuad <- cmQuad$byClass['Pos Pred Value']    
recallQuad <- cmQuad$byClass['Sensitivity']
f_measureQuad <- 2 * ((precisionQuad * recallQuad) / (precisionQuad + recallQuad))
quadmetrics = c(precisionQuad,recallQuad,f_measureQuad)

svmRbf <- svm(Xtrain,Ytrain,type = 'C-classification',kernel='radial',gamma = g)
YpredRbf = predict(svmRbf,Xtest)
cmRbf<-confusionMatrix(Ytest,YpredRbf)
precisionRbf <- cmRbf$byClass['Pos Pred Value']    
recallRbf <- cmRbf$byClass['Sensitivity']
f_measureRbf <- 2 * ((precisionRbf * recallRbf) / (precisionRbf + recallRbf))
rbfMetrics = c(precisionRbf,recallRbf,f_measureRbf)

svmSigmoid <- svm(Xtrain,Ytrain,type = 'C-classification',kernel='sigmoid',gamma = g,coeff0 = 0)
YpredSigmoid = predict(svmSigmoid,Xtest)
cmSigmoid<-confusionMatrix(Ytest,YpredSigmoid)
precisionSigmoid <- cmSigmoid$byClass['Pos Pred Value']    
recallSigmoid <- cmSigmoid$byClass['Sensitivity']
f_measureSigmoid <- 2 * ((precisionSigmoid * recallSigmoid) / (precisionSigmoid + recallSigmoid))
sigmoidMetrics = c(precisionSigmoid,recallSigmoid,f_measureSigmoid)


#all Evaluation metrics in tabular form
result <- as.matrix(rbind(linearMetrics,polyMetrics,quadmetrics,rbfMetrics,sigmoidMetrics))

colnames(result) <- c("Precision","Recall","F-Measure")
rownames(result) <- c("linear","polynomial","quadratic","radial","sigmoid")

print(result)
#View(result)



