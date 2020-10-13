#This module contains various text classifiers that can be used on a provided text dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import nltk
import tensorflow as tf
import keras
import sklearn
import csv
import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import operator
import seaborn as sns
import pickle

## This function is called by the classifiers to provide a visulization of the results
class_names = ['False', 'True']
def plot_confusion_matrix(cm, classes, Y_test,
                          predictions,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print("Confusion Matrix:")
    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j],'d'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    print(classification_report(Y_test, predictions, target_names = classes))

#### Setup the google drive connection if needed
##from google.colab import drive
##drive.mount('/content/gdrive')

### Enter filename below:
file_in = "/content/gdrive/My Drive/Colab Notebooks/various_data/praise/praise_data_multisemester_cleaned.csv"
data_col = "REVIEW"
label_col = "TAG"

### For testing on other datasets (cross-domain)
file_in2 = "/content/gdrive/My Drive/Colab Notebooks/amazon_sentiment_reviews_cleaned.csv"
data_col2 = "REVIEW"
label_col2 = "TAG"

df = pd.read_csv(file_in, engine = 'python')
df = df.dropna()
df = df.reset_index()
df = df.drop(columns = ['index'])
df2 = pd.read_csv(file_in2, engine = 'python') # Other
df2 = df2.dropna()
df2 = df2.reset_index()
df2 = df2.drop(columns = ['index'])
print(file_in)
print(df.dtypes)
print("Sample size:", len(df))
df.head(5)

### Define data and label columns
X = df[data_col] # Main dataset text
Y = df[label_col] # Main dataset labels
X2 = df2[data_col2] # Transfer text
Y2 = df2[label_col2] # Transfer labels

### Train and Test splitting
X_train, X_test, Y_train, Y_test = train_test_split(
 X, Y, test_size=0.15, random_state=42, stratify=df[label_col])
print("Train data amount:", len(X_train))
print("Test data amount:", len(X_test))

# Logistic  Regression Classifier

#### Logistic Regression
text_clf_log = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
 ('tfidf', TfidfTransformer()),
 ('clf-log', LogisticRegression(solver='liblinear')),
])

text_clf_log = text_clf_log.fit(X_train,Y_train)
### Function that shows must important features by class (binary)
def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

show_most_informative_features(text_clf_log['vect'], text_clf_log['clf-log'], n=10)

predicted_log = text_clf_log.predict(X_test)
accuracy = np.mean(predicted_log == Y_test)
print("Accuracy:", accuracy)
cm = confusion_matrix(Y_test, predicted_log)
print(cm)

## Graphical visualization
cnf_matrix = confusion_matrix(Y_test, predicted_log)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      Y_test = Y_test, predictions = predicted_log,
                      title='Confusion matrix')


#Gridsearch: Logistic Regression

#### Gridsearch
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              #'tfidf__use_idf': (True, False),
              'clf-log__C': (10, 1, 1e-1),
              'clf-log__solver': ('liblinear', 'newton-cg', 'lbfgs'),
    }

text_clf_log.get_params().keys()

## Cross-validation and fit
gs_clf_log = GridSearchCV(text_clf_log, parameters, cv=5, n_jobs=-1)
gs_clf_log = gs_clf_log.fit(X_train,Y_train)
predicted_gs = gs_clf_log.predict(X_test)
accuracy = np.mean(predicted_gs == Y_test)

print("Grid search best score:", gs_clf_log.best_score_)
print(gs_clf_log.best_params_)
print("Accuracy:", accuracy)

## Graphical visualization
cnf_matrix = confusion_matrix(Y_test, predicted_gs)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      Y_test = Y_test, predictions = predicted_gs,
                      title='Confusion matrix')

###### Transfer Learning Results ######
cnf_matrix = confusion_matrix(Y2, gs_clf_log.predict(X2))
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      Y_test = Y2, predictions = gs_clf_log.predict(X2),
                      title='Confusion matrix')



#### Random Forest
text_clf_rfc = Pipeline([('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
#  ('clf-rfc', RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42)),
 ('clf-rfc', RandomForestClassifier(n_estimators=300, max_depth=100)),
])

text_clf_rfc = text_clf_rfc.fit(X_train,Y_train)