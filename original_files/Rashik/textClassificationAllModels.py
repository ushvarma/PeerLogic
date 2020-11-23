# encoding=utf8
import csv
from random import seed
from random import randrange
import re
import pandas
import xgboost
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.ensemble import AdaBoostClassifier


lemmatizer = WordNetLemmatizer()

labels, texts = [], []

train_x = list()
valid_x = list()
train_y = list()
valid_y = list()


def train_and_predict(model, train_vector, label, test_vector, is_neural_net=False):
    model.fit(train_vector, label)
    predictions = model.predict(test_vector)

    # print predictions
    # print valid_y

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

#    print "Classification Metrics are:"
#    print(classification_report(valid_y, predictions))
    return metrics.accuracy_score(valid_y, predictions)

# f = open('new_data.csv', 'w')

# Data Preprocessing . Very Important
with open('new_problem_detection_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 1
    for row in csv_reader:
        # print line_count
        line_count = line_count + 1
        result = row[0].lower().strip()
        result = re.sub(r'\d+', '', result)
        result = result.translate(string.maketrans("", ""), string.punctuation)
        input_str = word_tokenize(result)
        comment = ""

        #  to remove nouns, pronouns from text use the below loop else comment it and use the loop after it
        # s = ' '
        # line = s.join(input_str)
        # blob = TextBlob(line)
        # for word, pos in blob.tags:
        #     # print word, pos
        #     if pos == "NN" or pos == "NNPS" or pos == "PRP" or pos == "PRP$":
        #         continue
        #     else:
        #         w = lemmatizer.lemmatize(word)
        #         comment = comment + " " + w

        for word in input_str:
            # print(word)
            w = lemmatizer.lemmatize(word)
            # print w
            comment = comment + " " + w
            # to remove stopwords uncomment below lines and comment the above 2 lines
            # removing stop words decreases the accuracy of almost all models, so not a good idea
            # if word not in stopwords.words('english'):
            #     w = lemmatizer.lemmatize(word)
            #     # print w
            #     comment = comment + " " + w

        # print(comment.strip())

        if comment != "":
            texts.append(comment.strip())
            # f.write(comment.strip() + "\n")
            labels.append(row[1])

# f.close()

# Split a dataset into k folds
folds = 4
dataset_split1 = texts
dataset_new1 = list()
dataset_copy1 = list(texts)
dataset_split2 = labels
dataset_new2 = list()
dataset_copy2 = list(labels)
fold_size = int(len(texts) / folds)
for i in range(folds):
    fold1 = list()
    fold2 = list()
    while len(fold1) < fold_size:
        index = randrange(len(dataset_copy1))
        fold1.append(dataset_copy1.pop(index))
        fold2.append(dataset_copy2.pop(index))
    dataset_new1.append(fold1)
    dataset_new2.append(fold2)
    # return dataset_split

# test cross validation split
seed(1)
folds1 = dataset_new1
folds2 = dataset_new2

for i in range(4):
    train_x = list()
    valid_x = list()
    train_y = list()
    valid_y = list()

    for j in range(4):
        if j == i:
            valid_x.extend(folds1[j])
            valid_y.extend((folds2[j]))
        else:
            train_x.extend(folds1[j])
            train_y.extend((folds2[j]))

    trainDF = pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels

    temp_y = []
    for a in train_y:
        temp_y.append(int(a))

    train_y = temp_y

    temp_y = []
    for b in valid_y:
        temp_y.append(int(b))

    valid_y = temp_y

    # Creating Count Vector //token_pattern is to consider words of more than 1 char
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(trainDF['text'])

    # transforming data for count vecotr
    count_train_vect = count_vect.transform(train_x)
    count_class_vect = count_vect.transform(valid_x)

    # Creating Word Level Tf-idf //token_pattern is to consider words of more than 1 char
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=50000)
    tfidf_vect.fit(trainDF['text'])
    tfidf_train = tfidf_vect.transform(train_x)
    tfidf_class = tfidf_vect.transform(valid_x)

    # Creating NGram Level Tf-idf // here taking vocabulary of 2 to 4 words.
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 4), max_features=50000)
    tfidf_vect_ngram.fit(trainDF['text'])
    tfidf_ngram_train = tfidf_vect_ngram.transform(train_x)
    tfidf_ngram_class = tfidf_vect_ngram.transform(valid_x)

    # Creating Character Level Tf-idf // here taking vocabulary of words made up of 2 to 10 characters
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 10),
                                             max_features=100000)
    tfidf_vect_ngram_chars.fit(trainDF['text'])
    tfidf_ngram_train_chars = tfidf_vect_ngram_chars.transform(train_x)
    tfidf_ngram_class_chars = tfidf_vect_ngram_chars.transform(valid_x)

    # Linear Classifier
    accuracy = train_and_predict(linear_model.LogisticRegression(solver='lbfgs'), count_train_vect, train_y,
                                 count_class_vect)
    print("Linear Classifier using Count Vectors: ", accuracy)

    accuracy = train_and_predict(linear_model.LogisticRegression(solver='lbfgs'), tfidf_train, train_y, tfidf_class)
    print("Linear Classifier using Word Level Tf-idf: ", accuracy)

    accuracy = train_and_predict(linear_model.LogisticRegression(solver='lbfgs'), tfidf_ngram_train, train_y,
                                 tfidf_ngram_class)
    print("Linear Classifier using NGram Tf-idf: ", accuracy)

    accuracy = train_and_predict(linear_model.LogisticRegression(solver='lbfgs'), tfidf_ngram_train_chars, train_y,
                                 tfidf_ngram_class_chars)
    print("Linear Classifier using Character Tf-idf Vectors: ", accuracy)

    # Naive Bayes
    accuracy = train_and_predict(naive_bayes.MultinomialNB(), count_train_vect, train_y, count_class_vect)
    print("Naive Bayes using Count Vectors: ", accuracy)

    accuracy = train_and_predict(naive_bayes.MultinomialNB(), tfidf_train, train_y, tfidf_class)
    print("Naive Bayes using Word Level Tf-idf: ", accuracy)

    accuracy = train_and_predict(naive_bayes.MultinomialNB(), tfidf_ngram_train, train_y, tfidf_ngram_class)
    print("Naive Bayes using NGram Tf-idf: ", accuracy)

    accuracy = train_and_predict(naive_bayes.MultinomialNB(), tfidf_ngram_train_chars, train_y, tfidf_ngram_class_chars)
    print("Naive Bayes using Character Tf-idf Vectors: ", accuracy)

    # SVM
    accuracy = train_and_predict(svm.SVC(gamma='scale'), count_train_vect, train_y, count_class_vect)
    print("SVM using Count Vectors: ", accuracy)

    accuracy = train_and_predict(svm.SVC(gamma='scale'), tfidf_train, train_y, tfidf_class)
    print("SVM using Word Level Tf-idf: ", accuracy)

    accuracy = train_and_predict(svm.SVC(gamma='scale'), tfidf_ngram_train, train_y, tfidf_ngram_class)
    print("SVM using NGram Tf-idf: ", accuracy)

    accuracy = train_and_predict(svm.SVC(gamma='scale'), tfidf_ngram_train_chars, train_y, tfidf_ngram_class_chars)
    print("SVM using Character Tf-idf Vectors: ", accuracy)

    # RandomForest (Bagging) //n_estimators are number of trees in forest
    accuracy = train_and_predict(ensemble.RandomForestClassifier(n_estimators=100), count_train_vect, train_y,
                                 count_class_vect)
    print("RandomForest using Count Vectors: ", accuracy)

    accuracy = train_and_predict(ensemble.RandomForestClassifier(n_estimators=100), tfidf_train, train_y, tfidf_class)
    print("RandomForest using Word Level Tf-idf: ", accuracy)

    accuracy = train_and_predict(ensemble.RandomForestClassifier(n_estimators=100), tfidf_ngram_train, train_y,
                                 tfidf_ngram_class)
    print("RandomForest using NGram Tf-idf: ", accuracy)

    accuracy = train_and_predict(ensemble.RandomForestClassifier(n_estimators=100), tfidf_ngram_train_chars, train_y,
                                 tfidf_ngram_class_chars)
    print("RandomForest using Character Tf-idf Vectors: ", accuracy)

    # Extreme Gradient Boosting
    accuracy = train_and_predict(xgboost.XGBClassifier(), count_train_vect, train_y, count_class_vect)
    print("Extreme Gradient Boosting using Count Vectors: ", accuracy)

    accuracy = train_and_predict(xgboost.XGBClassifier(), tfidf_train, train_y, tfidf_class)
    print("Extreme Gradient Boosting using Word Level Tf-idf: ", accuracy)

    accuracy = train_and_predict(xgboost.XGBClassifier(), tfidf_ngram_train, train_y, tfidf_ngram_class)
    print("Extreme Gradient Boosting using NGram Tf-idf: ", accuracy)

    accuracy = train_and_predict(xgboost.XGBClassifier(), tfidf_ngram_train_chars, train_y,
                                 tfidf_ngram_class_chars)
    print("Extreme Gradient Boosting using Character Tf-idf Vectors: ", accuracy)

    # Adaboost Classifier
    accuracy = train_and_predict(AdaBoostClassifier(n_estimators=100, learning_rate=1), count_train_vect, train_y,
                                 count_class_vect)
    print("Adaboost Classifier using Count Vectors: ", accuracy)

    accuracy = train_and_predict(AdaBoostClassifier(n_estimators=100, learning_rate=1), tfidf_train, train_y, tfidf_class)
    print("Adaboost Classifier using Word Level Tf-idf: ", accuracy)

    # Increasing the n_estimators, increase the time these 2 below statements take
    # accuracy = train_and_predict(AdaBoostClassifier(n_estimators=100, learning_rate=1), tfidf_ngram_train, train_y,
    #                              tfidf_ngram_class)
    # print("Adaboost Classifier using NGram Tf-idf: ", accuracy)
    #
    # accuracy = train_and_predict(AdaBoostClassifier(n_estimators=100, learning_rate=1), tfidf_ngram_train_chars, train_y,
    #                              tfidf_ngram_class_chars)
    # print("Adaboost Classifier using Character Tf-idf Vectors: ", accuracy)


    # Neural Net is giving very less accuracy. So not good for problem detection
#    def neural_net_creation(size):
#        # Input Layer
#        input_layer = layers.Input((size,), sparse=True)
#        
#        # Hidden Layer
#        hidden_layer = layers.Dense(100, activation="relu")(input_layer)
#        
#        # Output Layer
#        output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)
#        
#        classifier = models.Model(inputs=input_layer, outputs=output_layer)
#        classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
#        return classifier
#
#
#    classifier = neural_net_creation(count_train_vect.shape[1])
#    accuracy = train_and_predict(classifier, count_train_vect, train_y, count_class_vect, is_neural_net=True)
#    print("Neural Net using Count Vector: ", accuracy)
#
#    classifier = neural_net_creation(tfidf_train.shape[1])
#    accuracy = train_and_predict(classifier, tfidf_train, train_y, tfidf_class, is_neural_net=True)
#    print("Neural Net using Word Tf-idf: ", accuracy)
#
#    classifier = neural_net_creation(tfidf_ngram_train.shape[1])
#    accuracy = train_and_predict(classifier, tfidf_ngram_train, train_y, tfidf_ngram_class, is_neural_net=True)
#    print("Neural Net using NGram Tf-idf: ", accuracy)
#
#    classifier = neural_net_creation(tfidf_ngram_train_chars.shape[1])
#    accuracy = train_and_predict(classifier, tfidf_ngram_train_chars, train_y, tfidf_ngram_class_chars, is_neural_net=True)
#    print("Neural Net using Character Tf-idf: ", accuracy)

    print("###################################################################################################")
