#!/usr/bin/python

# encoding=utf8
from random import seed
from random import randrange
import csv
import re
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

X = []
Y = []
lemmatizer = WordNetLemmatizer()


# Implementing Multinomial Naive Bayes from scratch
class MultinomialNaiveBayes:

    def __init__(self):
        self.count = {}
        self.classes = None

    def fit(self, train_list, train_class_list):
        # print(len(train_list))
        # print(len(train_class_list))

        self.classes = set(train_class_list)
        for class_ in self.classes:
            self.count[class_] = {}
            for i in range(len(train_list[0])):
                self.count[class_][i] = 0
            self.count[class_]['total'] = 0
        self.count['total_points'] = len(train_list)

        # print self.count
        # print "#########"
        # print train_list

        for i in range(len(train_list)):
            for j in range(len(train_list[0])):
                self.count[train_class_list[i]][j] += train_list[i][j]
            self.count[train_class_list[i]]['total'] += 1

    def calculate_prob(self, test_point, class_):
        # print("here ", len(self.count))
        # log_prob = np.log(self.count[class_]['total']) - np.log(self.count['total_points'])
        log_prob = np.log(self.count[class_]['total'])  # # modified the formula, gives better accuracy

        total_words = len(test_point)
        for i in range(len(test_point)):
            # print(self.count[class_])
            # current_word_prob = test_point[i] * (
            #             np.log(self.count[class_][i] + 1) - np.log(self.count[class_]['total'] + total_words))

            current_word_prob = test_point[i] * (np.log(self.count[class_][i] + 1))  # modified the formula, gives better accuracy
            log_prob += current_word_prob

        return log_prob

    def predict_class_for_one(self, test_point):

        best_class = None
        best_prob = None
        first_run = True

        for class_ in self.classes:
            log_probability_current_class = self.calculate_prob(test_point, class_)
            # print("here3 ", log_probability_current_class)
            # print("here4", class_)
            if (first_run) or (log_probability_current_class > best_prob):
                best_class = class_
                best_prob = log_probability_current_class
                first_run = False

        return best_class

    def predict(self, test_list):
        test_predictions = []
        print(len(test_list))
        for i in range(len(test_list)):
            # print(test_list[i])
            test_predictions.append(self.predict_class_for_one(test_list[i]))

        return test_predictions

    def score(self, test_predictions, true_test_classes):
        # returns the mean accuracy
        count = 0
        for i in range(len(test_predictions)):
            # print(test_predictions[i], " , ", true_test_classes[i])
            if test_predictions[i] == true_test_classes[i]:
                count += 1
        # print(count)
        # print(len(test_predictions))
        # q = count / len(test_predictions)
        # print(q)
        x = 0.0 + len(test_predictions)   # to make float type
        y = 0.0 + count
        return y / x


# with open('expertiza_new_clean_data.csv') as csv_file:
with open('new_problem_detection_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        # print line_count
        # line_count = line_count + 1
        result = row[0].lower().strip()
        result = re.sub(r'\d+', '', result)
        result = result.translate(string.maketrans("", ""), string.punctuation)
        input_str = word_tokenize(result)
        comment = ""
        for word in input_str:
            # w = lemmatizer.lemmatize(word.decode("utf8", "ignore"))
            # print(word)
            w = lemmatizer.lemmatize(word)
            # print w
            comment = comment + " " + w

        # print(comment.strip())

        X.append(comment.strip())
        Y.append(row[1])


# Split a dataset into k folds
folds = 4
dataset_split1 = X
dataset_new1 = list()
dataset_copy1 = list(X)
dataset_split2 = Y
dataset_new2 = list()
dataset_copy2 = list(Y)
fold_size = int(len(X) / folds)
for i in range(folds):
    fold1 = list()
    fold2 = list()
    while len(fold1) < fold_size:
        index = randrange(len(dataset_copy1))
        fold1.append(dataset_copy1.pop(index))
        fold2.append(dataset_copy2.pop(index))
    dataset_new1.append(fold1)
    dataset_new2.append(fold2)

seed(1)
folds1 = dataset_new1
folds2 = dataset_new2
# print(folds1)
# print(folds2)


for i in range(4):
    train_list = list()
    test_list = list()
    train_class_list = list()
    test_class_list = list()

    for j in range(4):
        if j == i:
            test_list.extend(folds1[j])
            test_class_list.extend((folds2[j]))
        else:
            train_list.extend(folds1[j])
            train_class_list.extend((folds2[j]))

    # print train_list

    stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone',
                 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount',
                 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around',
                 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before',
                 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both',
                 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de',
                 'describe', 'detail', 'did', 'do', 'does', 'doing', 'don', 'done', 'down', 'due', 'during', 'each', 'eg',
                 'eight', 'either', 'eleven', 'else', 'els.decode("utf8")ewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every',
                 'everyone',
                 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five',
                 'for',
                 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go',
                 'had',
                 'has', 'hasnt', 'have', 'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein',
                 'hereupon',
                 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc',
                 'indeed',
                 'interest', 'into', 'is', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least',
                 'less',
                 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most',
                 'mostly',
                 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next',
                 'nine',
                 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on',
                 'once',
                 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over',
                 'own',
                 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed', 'seeming',
                 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so',
                 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system',
                 't', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence',
                 'there',
                 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third',
                 'this',
                 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top',
                 'toward',
                 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was',
                 'we',
                 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas',
                 'whereby',
                 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole',
                 'whom',
                 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself',
                 'yourselves']

    # Create words_count from train_list
    words_count = {}
    for i in range(len(train_list)):
        word_list = []
        for word in train_list[i].split():
            word_new = word.strip(string.punctuation).lower()
            # Removing stop words is actually decreasing the accuracy of Multinomial model for problem detection
            # uncomment the below line and comment the line after it to remove stop words
            # if (len(word_new) > 2) and (word_new not in stopwords):
            if len(word_new) > 0:
                if word_new in words_count:
                    words_count[word_new] += 1
                else:
                    words_count[word_new] = 1

    # Plotting a graph of no of words with a given frequency to decide cutoff frequency
    num_words = [0 for i in range(max(words_count.values()) + 1)]
    freq = [i for i in range(max(words_count.values()) + 1)]
    for key in words_count:
        num_words[words_count[key]] += 1
    # plt.plot(freq, num_words)
    # plt.axis([1, 10, 0, 20000])
    # plt.xlabel("Frequency")
    # plt.ylabel("No of words")
    # plt.grid()
    # plt.show()

    print("total significant words are:", len(words_count))

    # Setting cutoff frequency right is very important. Based on the plots I came up with above,
    # I figured setting it 10 gives best accuracy for training and test both
    cutoff_freq = 10

    num_words_above_cutoff = len(words_count) - sum(num_words[0:cutoff_freq])
    print("Number of words with frequency higher than cutoff frequency({}) :".format(cutoff_freq), num_words_above_cutoff)

    # We consider words with frequency higher than cutoff freq
    features = []
    for key in words_count:
        if words_count[key] >= cutoff_freq:
            features.append(key)

    # To represent training data as word vector counts
    train_list_dataset = np.zeros((len(train_list), len(features)))

    for i in range(len(train_list)):
        # print(i)
        word_list = [word.strip(string.punctuation).lower() for word in train_list[i].split()]
        for word in word_list:
            if word in features:
                train_list_dataset[i][features.index(word)] += 1

    # To represent test data as word vector counts
    test_list_dataset = np.zeros((len(test_list), len(features)))
    for i in range(len(test_list)):
        # print(i)
        word_list = [word.strip(string.punctuation).lower() for word in test_list[i].split()]
        for word in word_list:
            if word in features:
                test_list_dataset[i][features.index(word)] += 1


    # Using sklearn's Multinomial Naive Bayes
    clf = MultinomialNB()
    clf.fit(train_list_dataset, train_class_list)
    test_class_list_pred = clf.predict(test_list_dataset)
    sklearn_score_train = clf.score(train_list_dataset, train_class_list)
    print("Sklearn's score on training data :", sklearn_score_train)
    sklearn_score_test = clf.score(test_list_dataset, test_class_list)
    print("Sklearn's score on testing data :", sklearn_score_test)
    print("Classification report for testing data :-")
    print(classification_report(test_class_list, test_class_list_pred))
    # print("***********************************************************")

    # Using our own Multinomial Model
    clf2 = MultinomialNaiveBayes()
    clf2.fit(train_list_dataset, train_class_list)
    test_class_list_pred = clf2.predict(test_list_dataset)
    our_score_test = clf2.score(test_class_list_pred, test_class_list)
    print("Our score on testing data :", our_score_test)
    print("Classification report for testing data from our model:-")
    print(classification_report(test_class_list, test_class_list_pred))

    print("Score of our model on test data:", our_score_test)
    print("Score of inbuilt sklearn's MultinomialNB on the same data :", sklearn_score_test)
    print("***********************************************************")

# Help taken from https://github.com/hmahajan99/Text-Classification/blob/master/Text%20Classification%20Using%20Naive%20Bayes.py
