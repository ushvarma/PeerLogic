import nltk
import csv
import io
from string import digits
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from collections import Counter
from decimal import *
import pandas as pd
import numpy
import sys
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')

getcontext().prec = 3

sid = SentimentIntensityAnalyzer()

sen_list_with_problem = list()
sen_list_without_problem = list()

def getSentimentScore(sentence, senti):
	temp = sentence.split('.')
	count_temp = 0.0
	for sen in temp:		
		count_temp += sid.polarity_scores(sen)[senti]
	print (count_temp/len(temp))
	return count_temp/len(temp)

list1 = list()
list2 = list()
vectorizer = 2
X = 2
def createTfIDFFeature(list1, list2):
	global vectorizer
	global X
	doc1 = ' '.join(list1)
	doc2 = ' '.join(list2)
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform([doc1, doc2])


def getIndex(word_test):
	global vectorizer
	global X
	index = 0
	for word in vectorizer.get_feature_names():
		if word == word_test:
			return index
		index += 1
	return -1

def getSenScore(sen):
	global vectorizer
	global X
	score_class1 = 0
	score_class2 = 0
	for word in sen.split(' '):
		index = getIndex(word)
		if index != -1:
			score_class1 += X[0, index]
			score_class2 += X[1, index]
	return [score_class1, score_class2]

#createTfIDFFeature(sen_list_with_problem, sen_list_without_problem)
#df = pd.read_csv("train_set2.csv")

def createScores():
	df = pd.read_csv("train_set3.csv")
	for i in range(0, df.shape[0]):
		print ("done" + str(i) + "/" + str(df.shape[0]))
		sen = ''.join(x for x in df.comments[i] if x.isalpha() or x ==' ')
		tokens = nltk.word_tokenize(sen.lower())
		text = nltk.Text(tokens)
		tags = nltk.pos_tag(text)
		NN_count = 0.0
		VB_count = 0.0
		AD_count= 0.0
		ADV_count = 0.0
		counts = Counter(tag for word,tag in tags)
		tot = sum(counts.values())
		for ele in counts:
			if ele == 'NN' or ele == 'NNP' or ele == 'NNS':
				NN_count += counts[ele]
			if ele == 'RB' or ele == 'RBR' or ele == 'RBS':
				ADV_count += counts[ele]
			if ele == 'VB' or ele == 'VBD' or ele == 'VBG' or ele == 'VBN' or ele == 'VBP' or ele == 'VBZ':
				VB_count += counts[ele]
			if ele == 'JJ' or ele == 'JJR' or ele == 'JJS':
				AD_count += counts[ele]
		if tot != 0:
			df.NN[i] = round(NN_count/tot, 2)
			df.RB[i] = round(VB_count/tot, 2)
			df.VB[i] = round(AD_count/tot, 2)
			df.JJ[i] = round(ADV_count/tot, 2)
	df.to_csv('train_set4.csv', index=False)
createScores()

def cor_test():
	for i in range(0, df.shape[0]):
	#for i in range(0, 5):
		print ('done: ' + str(i) + '/' + str(df.shape[0]))
		df.neg_senti[i] = getSentimentScore(df.comments[i], 'neg') 
		df.pos_senti[i] = getSentimentScore(df.comments[i], 'pos')
		res = getSenScore(df.comments[i])
		df.tf_score[i] = -1 if res[0]< res[1] else 1
	df.to_csv('train_set2.csv', index=False)

	res = 0.0
	cor_res = 0.0
	for i in range(0, df.shape[0]):
		if df.tf_score[i].astype(numpy.int64) == df.value[i]:
			cor_res += 1.0
		res += 1.0

	print (cor_res/res, res, cor_res)


def sentimentScoreAttributeAnalysis():
	ans1 = 0
	ans2 = 0

	for i in range(0, len(sen_list_with_problem)):
		temp = sen_list_with_problem[i].split('.')
		count_temp = 0
		for sen in temp:		
			count_temp += sid.polarity_scores(sen)['neg']
		ans1 += count_temp/len(temp)
		list1.append(count_temp)
	plt.scatter([0 for i in range(0, len(list1))], list1, color = "blue", label='with problems')

	for i in range(0, len(sen_list_without_problem)):
		temp = sen_list_without_problem[i].split('.')
		count_temp = 0
		for sen in temp:		
			count_temp += sid.polarity_scores(sen)['neg']
		ans2 += count_temp/len(temp)
		list2.append(count_temp)
	plt.scatter([0.2 for i in range(0, len(list2))], list2, color = "red", label='without problems')

	print ("Mean NEG value for Sen with problems: "), ans1/len(sen_list_with_problem)
	print ("Mean NEG value for Sen without problems: "), ans2/len(sen_list_without_problem)

	ans1 = 0
	ans2 = 0
	for i in range(0, len(sen_list_with_problem)):
		temp = sen_list_with_problem[i].split('.')
		count_temp = 0
		for sen in temp:		
			count_temp += sid.polarity_scores(sen)['pos']
		ans1 += count_temp/len(temp)
		list1.append(count_temp)
	plt.scatter([2 for i in range(0, len(list1))], list1, color = "blue")

	for i in range(0, len(sen_list_without_problem)):
		temp = sen_list_without_problem[i].split('.')
		count_temp = 0
		for sen in temp:
			count_temp += sid.polarity_scores(sen)['pos']
		ans2 += count_temp/len(temp)
		list2.append(count_temp)
	plt.scatter([2.2 for i in range(0, len(list2))], list2, color = "red")
	plt.legend(loc='upper right')
	plt.show()

	print ("Mean POS value for Sen with problems: ", ans1/len(sen_list_with_problem))
	print ("Mean POS value for Sen without problems: ", ans2/len(sen_list_without_problem))

