# Read Json Files
## This module handles the process of reading through json data and storing it into a pandas dataframe to convert into a csv file.

### Import the required libraries.
import json
import pandas as pd
import csv
from bs4 import BeautifulSoup

##from google.colab import drive
##drive.mount('/content/gdrive')

file_in = "/content/gdrive/My Drive/Colab Notebooks/various_json/localize/new_localize_data_original.json"

### The code below parses the json file and stores data from each attribute into a list.

ans_ids = []
tags = []
reviews = []
tag_descs = []
taggers = []
crt_times = []
updt_times = []
with open(file_in) as json_file:
    data = json.load(json_file)
    inc = 0
    for p in data:
        ans_ids.append(p['ANS_ID'])
        tags.append(p['TAG'])
        soup = BeautifulSoup(p['REVIEW'])
        text = soup.get_text()
        reviews.append(text)
        tag_descs.append(p['TAG_DESC'])
        taggers.append(p['TAGGER'])
        crt_times.append(p['CRT_TIME'])
        updt_times.append(p['UPDT_TIME'])

### This code creates a pandas dataframe out of the parsed json data.
new_data = pd.DataFrame({
                         "ANS_ID": ans_ids,
                         "TAG": tags,
                         "REVIEW": reviews,
                         "TAG_DESC": tag_descs,
                         "TAGGER": taggers,
                         "CRT_TIME": crt_times,
                         "UPDT_TIME": updt_times,
                         })
new_data.head(5) # Shows sample of the new dataset

file_out = "/content/gdrive/My Drive/Colab Notebooks/various_json/localize/new_localize_data_parsed.csv"

new_data.to_csv(file_out, index=False)