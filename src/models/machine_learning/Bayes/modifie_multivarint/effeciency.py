from sys import argv

script, file1, file2 = argv

with open(file1) as f:
    lines1 = f.readlines()

with open(file2) as f:
    lines2 = f.readlines()

match = 0

for i in range(len(lines2)):
    if lines2[i] == lines1[i]:
        match += 1

match_percentage = (float(match) * 100) / float(len(lines1))
print "Documents matched : " + str(match)
print len(lines1)
print "Percentage match : " + str(match_percentage)
