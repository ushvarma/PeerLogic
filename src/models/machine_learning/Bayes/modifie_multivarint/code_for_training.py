import mysql.connector
# import MySQLdb
from sys import argv

# take the name of the file containing the training data from command line
script, filename = argv

target = open(filename, 'r')

lines = target.readlines()
class_prob = {}
count_docs = {}
vocab = []
i = 1
print "Calculating P(class)"
for line in iter(lines):
    word = line.split()
    # print line
    if word[0] in class_prob:
        class_prob[word[0]] += 1.0
        count_docs[word[0]] += 1
    else:
        class_prob[word[0]] = 1.0
        count_docs[word[0]] = 1
    vocab = vocab + word[1:]
    # print i
    i += 1

for key in class_prob:
    class_prob[key] = class_prob[key] / len(lines)

print "Extracting vocabulary"
# list() used to remove duplicate words
vocab = list(set(vocab))

print "Initializing dictionaries"
prob = {}
count = {}
for _word in vocab:
    prob[_word] = {}
    count[_word] = {}
    for _class in class_prob:
        prob[_word][_class] = 0.0
        count[_word][_class] = 0

print "Calculating P(word|class) and frequencies"
for _word in vocab:
    for _class in class_prob:
        for line in iter(lines):
            word = line.split()
            if word[0] == _class and _word in word:
                prob[_word][_class] += 1.0
                count[_word][_class] += word[1:].count(_word)
        prob[_word][_class] = (prob[_word][_class] + 1) / float(count_docs[_class] + 2)
        # print "P(" + _word + "|" + _class + ") = " + str(prob[_word][_class])
##################
# print "counts..."
# for _word in vocab:
#     for _class in class_prob:
#         print "count[" + _word + "|" + _class + "=" + str(count[_word][_class])
##################

print "Modifying"
print "Calculating means"
mean = {}
for _word in vocab:
    ct = 0
    for _class in class_prob:
        ct += count[_word][_class]
    mean[_word] = float(ct) / len(class_prob)
    # print "mean( " + _word + " ) = " + str(mean[_word])

sup = open("sup.txt", "w")
print "Final check step modification"
for _word in vocab:
    for _class in class_prob:
        # print "count(" + _word + "|" + _class + ") - mean(" + _word + ") >? 6*mean(" + _word + ") i.e. " + str(
            # float(count[_word][_class]) - mean[_word]) + ">" + str(6 * mean[_word])
        # if (float(count[_word][_class]) - mean[_word] > 6 * mean[_word]) and (count[_word][_class] > 10):
        if float(count[_word][_class]) > 30:
            temp_p = prob[_word][_class]
            prob[_word][_class] = prob[_word][_class] * count[_word][_class]
            # print "here***"
            sup.write("P(" + _word + "|" + _class + ") modified from " + str(temp_p) + " to " + str(
                prob[_word][_class]) + "\n")
sup.close()

print "Modified"
print "Inserting into database..."
db = mysql.connector.connect(
    user="root",
    password="rashik1994",
    host="localhost",
    database="naive"
)
cursor = db.cursor()
for _class in class_prob:
    sql = "insert into probability_class values('%s','%f')" % (_class, class_prob[_class])
    # print sql
    cursor.execute(sql)
    for _word in vocab:
        sql = 'insert into  probability_word_given_class values("%s","%s","%f")' % (_class, _word, prob[_word][_class])
        # print sql
        cursor.execute(sql)
db.commit()
db.close()
target.close()
