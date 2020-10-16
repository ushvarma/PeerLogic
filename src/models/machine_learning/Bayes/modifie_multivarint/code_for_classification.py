from sys import argv
from math import log
import mysql.connector
# import MySQLdb

db = mysql.connector.connect(
    user="root",
    password="rashik1994",
    host="localhost",
    database="naive"
)
cursor = db.cursor()

sql = "select * from probability_class"
cursor.execute(sql)
resultSet = cursor.fetchall()
class_prob = {}
for row in resultSet:
    class_prob[row[0]] = row[1]

sql = "select * from probability_word_given_class"
cursor.execute(sql)
resultSet = cursor.fetchall()
prob = {}
for row in resultSet:
    if row[1] not in prob:
        prob[row[1]] = {}
    prob[row[1]][row[0]] = row[2]


def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in line.split()]


def findClass(target):
    # target = list(set(target))
    max_prob = float("-inf")
    max_class = ""
    for _class in class_prob:
        _prob = log(class_prob[_class])
        for _word in target:
            if _word in prob:
                _prob += log(prob[_word][_class])
        if _prob > max_prob:
            max_prob = _prob
            max_class = _class

    return max_class
    db.close()


if __name__ == "__main__":
    script, filename = argv
    text = read_words(filename)
    findClass(text)
