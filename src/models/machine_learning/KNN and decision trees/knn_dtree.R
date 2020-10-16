require(caret)
require(rpart)

calculate_distance_matrix <- function(train_matrix, test_matrix, method_name){
  distance_matrix = matrix(0L, nrow = nrow(test_matrix), ncol = nrow(train_matrix))
    for(i in seq(1, nrow(test_matrix))){
      for(j in seq(1, nrow(train_matrix))){
        distance_matrix[i,j] <- do.call(method_name, list(unlist(test_matrix[i,]), unlist(train_matrix[j,])))
      }
    }
  return(distance_matrix)
}

calculate_euclidean <- function(p, q) {
  return (sqrt(sum((p - q) ^ 2)))
}

calculate_cosine <- function(p, q) {
  return (sum(p*q)/(sqrt(sum(p ^ 2)) * sqrt(sum(q ^ 2))))
}

knn_classifier <- function(x_train, y_train, x_test, distance_method, k){
  distance_matrix <- calculate_distance_matrix(x_train,x_test, distance_method)
  result<- vector()
  for(i in seq(1, nrow(distance_matrix))){
    vec <- vector()
    temp<-distance_matrix[i,]
    if(distance_method=='calculate_euclidean'){
      vec<-(sort(temp, index.return=TRUE)$ix)[1:k]
    }
    else if(distance_method=='calculate_cosine'){
      vec<-(sort(temp, index.return=TRUE,decreasing=TRUE)$ix)[1:k]
    }
    vec<-y_train[vec]
    result[i]<-strtoi(names(sort(table(vec),decreasing=TRUE)[1]))
  }
  return(as.factor(result))
}


knn_classifier_confidence <- function(x_train, y_train, x_test, distance_method='calculate_cosine', k){
  distance_matrix <- calculate_distance_matrix(x_train,x_test, distance_method)
  result<- vector()
  for(i in seq(1, nrow(distance_matrix))){
    vec <- vector()
    temp<-distance_matrix[i,]
    ordered<-(sort(temp, index.return=TRUE,decreasing=TRUE))[1:k]
    confidence<-sum(ordered$x[1:k])
    for(j in seq(1, k)){
      if(is.na(vec[toString(y_train[ordered$ix[j]])])){
        vec[toString(y_train[ordered$ix[j]])]<-0
      }
      vec[toString(y_train[ordered$ix[j]])]<-vec[toString(y_train[ordered$ix[j]])]+ordered$x[j]
    }
    vec<-vec/confidence
    result[i]<-strtoi(names(sort(vec,decreasing=TRUE)[1]))
  }
  return(as.factor(result))
  
}

dtree <- function(x_train, y_train, x_test){
  set.seed(123)
  x_train[,ncol(x_train)+1]<-y_train
  colnames(x_train)[ncol(x_train)] <- "Class"
  tree <- rpart(Class~.,
                data=x_train,
                method = "class",
                parms=list(split='gini'))
  plot(tree)
  text(tree)
  return(rpart_predict <- predict(tree,x_test,type="class"))
  
}


dtree_cv <- function(x_train, y_train, x_test, n_folds){
  set.seed(123)
  x_train[,ncol(x_train)+1]<-y_train
  colnames(x_train)[ncol(x_train)] <- "Class"
  
  train_control<- trainControl(method="cv", number=n_folds, savePredictions = TRUE)
  
  model<- train(Class~., data=x_train, trControl=train_control, method="rpart")
  
  cv_predict<- predict(model,x_test)
  
  return(cv_predict)
  
}


calculate_accuracy <- function(y_pred, y_true){
  cm <- as.matrix(table(Prediction = y_pred, Reference = y_true))
  n <- sum(cm) 
  diagnol <- diag(cm)
  accuracy <- sum(diagnol) / n 
  return(list(cm,accuracy))
}

