import pandas as pd
import csv
import re
from spellchecker import SpellChecker
import nltk
from nltk.corpus import words
import string

nltk.download('words')

spell = SpellChecker()

file_in = "../dataSets/all_data_multisemester_no_duplicates.csv"
file_out = "cleaned_dataset.csv"
data_col = "REVIEW"
label_col = "TAG"
df = pd.read_csv(file_in, engine = 'python');
print("Number of observations:", len(df))
df.head(5)

def preprocess_reviews(reviews, labels):
  spell = SpellChecker()
  print("Number of observations to parse:", len(reviews))
  comments = []
  tags = []
  for i in range(len(reviews)):
    if reviews[i] == "" or isinstance(reviews[i], str) == False or reviews[i] == " ":
            continue
    if i%1000 == 0:
        print("Update:", i)
    reviews[i] = re.sub(r'[!?]','.',reviews[i]) # Removing special character
    reviews[i] = re.sub(r'[^.a-zA-Z0-9\s]',' ',reviews[i]) # Removing special character
    reviews[i] = re.sub('\'',' ',reviews[i]) # Removing quotes
    reviews[i] = re.sub('#','',reviews[i]) # Removing quotes
    reviews[i] = re.sub('\d',' ',reviews[i]) # Replacing digits by space
    reviews[i] = re.sub(r'\s+[a-z][\s$]', ' ',reviews[i]) # Removing single characters and spaces alongside
    reviews[i] = re.sub(r'\s+', ' ',reviews[i]) # Replacing more than one space with a single space
    if 'www.' in reviews[i] or 'http:' in reviews[i] or 'https:' in reviews[i] or '.com' in reviews[i]:
          reviews[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", reviews[i])
    reviews[i] = reviews[i].lower()
    reviews[i] = reviews[i].rstrip()
    spot = reviews[i].find(' .')
    while spot != -1: # Fix lone periods in comment
      sl = list(reviews[i])
      sl[spot] = '.'
      sl[spot+1] = ''
      reviews[i] = "".join(sl)
      spot = reviews[i].find(' .')
    for word in reviews[i].split():
      if word == '.':
        continue
      word_base = word.translate(str.maketrans('', '', string.punctuation))  
      if(bool(spell.unknown([word_base]))):
        recommended = spell.correction(word_base)
        if (recommended in words.words()):
          reviews[i] = reviews[i].replace(word,recommended,1)
        else:
          reviews[i] = reviews[i].replace(word, '')
          reviews[i] = re.sub(r'\s+', ' ',reviews[i]) # Replacing more than one space with a single space
    reviews[i] = reviews[i].replace('..', '.')
    if reviews[i].find('.') == 0:
      reviews[i] = reviews[i].replace('.', '', 1)
      reviews[i] = reviews[i].replace(' ', '', 1)
    comments.append(reviews[i])
    tags.append(labels[i])
  return comments, tags

text = df[data_col]
labels = df[label_col]
text = text.tolist()
text, labels = preprocess_reviews(text, labels)

problems_data = pd.DataFrame({"TAG": labels,
                         "REVIEW": text,
                         })
print(problems_data.head(5))

problems_data.to_csv(file_out, index=False)