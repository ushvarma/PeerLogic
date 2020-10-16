rm(list=ls(all=T))
cat('\014')

source('./svm.R')

# set your working directory
# setwd()

# install all necessary packages
required_packages = c('e1071', 'caret', 'glmnet')
for(package in required_packages){
  if(!(package %in% installed.packages())){
    install.packages(package, dependencies = T)
  }   
}


# load the packages
library('e1071') # for SVM
library('glmnet') # for linear, ridge and lasso regression

# set a seed for reproducibility
set.seed(100)
############################################################################################################

load_data <- function(data_folder='./', learning_type){
  FullData <- read.csv(paste0(data_folder, learning_type, '-train.csv'), header=T)[1:2000,]
  
  smp_size <- floor(0.7 * nrow(FullData))
  
  ## set the seed to make your partition reproducible
  set.seed(123)
  train_ind <- sample(seq_len(nrow(FullData)), size = smp_size)
  
  train_df <- FullData[train_ind, ]
  test_df <- FullData[-train_ind, ]
  
  # make sure dependent variable is of type factor if this is classification
  if(learning_type == 'classification'){
    train_df$Class <- as.factor(train_df$Class)
    test_df$Class <- as.factor(test_df$Class)
  }
  print(train_df[,361])
  return(list(train_df, test_df))
}

##########################################################################################################

# load data necessary for classification
clf_data <- load_data(data_folder='./', learning_type='classification')
clf_train_df <- clf_data[[1]]
clf_test_df <- clf_data[[2]]

############################################################################################################ 
# Learning and parameter tuning 

########################################################
# Classification

# SVM

# linear kernel
linear_svm_result <- function_svm(x_train = clf_train_df[,-361], x_test = clf_test_df[,-361], y_train = clf_train_df[,361], 
                              kernel_name = 'linear')

# radial kernel
radial_svm_result <- function_svm(x_train = clf_train_df[,-361], x_test = clf_test_df[,-361], y_train = clf_train_df[,361], 
                              kernel_name = 'radial')

# sigmoid kernel
sigmoid_svm_result <- function_svm(x_train = clf_train_df[,-361], x_test = clf_test_df[,-361], y_train = clf_train_df[,361], 
                              kernel_name = 'sigmoid')

# polynomial kernel
# linear svm
polynomial_svm_result <- function_svm(x_train = clf_train_df[,-361], x_test = clf_test_df[,-361], y_train = clf_train_df[,361], 
                              kernel_name = 'polynomial')




# compare all classifiers
all_classifier_summary <- classification_compare_accuracy(y_test=clf_test_df[,361], 
                                                   linear_kernel_prediction = linear_svm_result[[2]], 
                                                   radial_kernel_prediction = radial_svm_result[[2]], 
                                                  polynomial_kernel_prediction = polynomial_svm_result[[2]], 
                                                  sigmoid_kernel_prediction = sigmoid_svm_result[[2]])

print(paste('Best classification model =', all_classifier_summary[[1]], 'Overall Accuracy =', all_classifier_summary[[2]]))







