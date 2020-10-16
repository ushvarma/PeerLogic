
calculate_rmse <- function(y_true, y_pred){
  print(sqrt(sum(y_true - y_pred)^2/length(y_true)))
  return(sqrt(sum(y_true - y_pred)^2/length(y_true)))
}

regression_compare_rmse <- function(y_test, linear_regression_prediction, ridge_prediction, lasso_prediction){
  rmse_vec<-c(calculate_rmse(y_test,linear_regression_prediction),calculate_rmse(y_test,ridge_prediction),calculate_rmse(y_test,lasso_prediction))
  rmse_types<-c('linear','ridge','lasso')
  index=which(rmse_vec==min(rmse_vec))
  return(list(rmse_types[index],rmse_vec[index],rmse_vec))
}


function_svm <- function(x_train, x_test, y_train, kernel_name){
  
   if(kernel_name == "radial"){
    # ~1-2 lines 
     prioir_svm <- tune.svm(x_train,y_train,cost = c(0.01, 0.1, 1, 10),gamma =c(0.05, 0.5, 1, 2))
     svmfit <-  svm(x_train,y_train,kernel="radial",cost=prioir_svm$best.parameters$cost,gamma = prioir_svm$best.parameters$gamma)
     
     
   }else if(kernel_name == 'polynomial'){
     #~1-2 lines
     prioir_svm <- tune.svm(x_train,y_train,cost = c(0.01, 0.1, 1, 10),gamma =c(0.05, 0.5, 1, 2),degree =c(1,2,3))
     svmfit <- svm(x_train,y_train,kernel="polynomial",cost=prioir_svm$best.parameters$cost,gamma = prioir_svm$best.parameters$gamma,degree =prioir_svm$best.parameters$degree)
     
   }else if(kernel_name == 'sigmoid'){
     #~1-2 lines
     prioir_svm <- tune.svm(x_train,y_train,cost = c(0.01, 0.1, 1, 10),gamma =c(0.05, 0.5, 1, 2))
     svmfit <- svm(x_train,y_train,kernel="sigmoid",cost=prioir_svm$best.parameters$cost,gamma = prioir_svm$best.parameters$gamma)
     
   }else{ # default linear kernel
     #~1-2 lines
     prioir_svm <- tune.svm(x_train,y_train,cost = c(0.01, 0.1, 1, 10))
     svmfit <- svm(x_train,y_train,kernel="linear",cost=prioir_svm$best.parameters$cost)
     
   }
  pred <- predict(svmfit,x_test)
  return(list(svmfit,pred))
}

calculate_accuracy <- function(y_true,y_pred){
  print(100*sum(as.vector(y_true)==as.vector(y_pred))/length(as.vector(y_true)))
  return (100*sum(as.vector(y_true)==as.vector(y_pred))/length(as.vector(y_true)))
}


classification_compare_accuracy <- function(y_test, linear_kernel_prediction, radial_kernel_prediction, 
                                            polynomial_kernel_prediction, sigmoid_kernel_prediction){

  accuracy_vec<-c(calculate_accuracy(y_test,linear_kernel_prediction),calculate_accuracy(y_test,radial_kernel_prediction)
              ,calculate_accuracy(y_test,polynomial_kernel_prediction),calculate_accuracy(y_test,sigmoid_kernel_prediction))
  accuracy_types<-c('svm-linear','svm-radial','svm-poly','svm-sigmoid')
  index=which(accuracy_vec==max(accuracy_vec))
  return(list(accuracy_types[index],accuracy_vec[index],accuracy_vec))
  
}

