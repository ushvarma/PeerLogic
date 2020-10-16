import code_for_classification
import string
from sys import argv

script, filename, correctoutput, myoutput = argv

target = open(filename, 'r')

lines = target.readlines()

target_correct = open(correctoutput, 'w')

target_my = open(myoutput, 'w')

i = 1
for line in iter(lines):
    words = [word for word in line.split()]
    target_my.write(code_for_classification.findClass(words[1:]) + "\n")
    target_correct.write(words[0] + "\n")
    # print(i)
    i += 1

target.close()
target_correct.close()
target_my.close()

