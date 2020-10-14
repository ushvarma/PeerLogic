#This module balances a dataset by minority class label 
#along with only including data where all taggers have agreed on a tag.

import pandas as pd
import csv
from preprocessing_utils import num_balanced_labels

file_in = "/content/gdrive/My Drive/Colab Notebooks/various_json/localize/new_localize_data_parsed.csv"
data_col = "REVIEW"
label_col = "TAG"
df = pd.read_csv(file_in, engine = 'python')
df.sort_values(by=['UPDT_TIME'], inplace=True, ascending=False) # Sort by timestamp
df = df.reset_index() # remove later
print("Number of samples:", len(df))
df.head(5)

num_labels = num_balanced_labels(data_col, label_col, df)
ls = []
lstag = []
idt = []
num_0 = 0 # Negative class label
num_1 = 0 # Positive class label
dropthese = [] # Indexes of comments to be dropped (comments without full inner-annotator agreement)
num_same = 0 # Number of times where a duplicate (annotated by multiple people) comment was tagged the same.
num_different = 0 # Number of times where a duplicate (annotated by multiple people) comment was not tagged the same.
print("Create: Balance observations")
for i in range(len(df)):
    if i%2000 == 0:
        print("Update:", i) # Tracks progress
    text = df[data_col][i]
    if text == "" or isinstance(text, str) == False or text == " ":
        continue
    text = text.strip()
    if text in ls:
        for j in range(len(idt)):
            if text == ls[j]:
                if df["TAG"][idt[j]] == df["TAG"][i]:
                    num_same = num_same + 1
                else:
                    num_different = num_different + 1
                    if df["TAGGER"][idt[j]] == df["TAGGER"][i]: # If tagger is the same, that means they are redoing a tag, so we use the most recent label.
                        upexis = df["UPDT_TIME"][idt[j]]
                        upcur = df["UPDT_TIME"][i]
                        if (upexis < upcur):
                            lstag[j] = df["TAG"][i]
                            newtag = df["TAG"][i] # change label balance counts
                            if newtag == 0:
                                num_1 = num_1 - 1
                                num_0 = num_0 + 1
                            else:
                                num_1 = num_1 + 1
                                num_0 = num_0 - 1
                    else: # If tagger is different, don't include this comment in the dataset.
                        dropthese[j] = 1
        continue
    val = int(df[label_col][i])
    if val == 1:
        num_1 = num_1 + 1
    else:
        num_0 = num_0 + 1
        val = 0 # Make sure negative label is 0 (ML libraries don't handle negative labels well)
    if val == 0 and (num_0 > num_labels):
        continue
    if val == 1 and (num_1 > num_labels):
        continue
    ls.append(text)
    lstag.append(val)
    idt.append(i)
    dropthese.append(0)
print("Number of times duplicate comments were tagged the same:", num_same)
print("Number of times duplicate comments were not tagged the same:", num_different)

file_out = "/content/gdrive/My Drive/Colab Notebooks/various_json/localize/new_localize_data_balanced.csv"

with open(file_out, 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow([label_col, data_col])
    for i in range(len(ls)):
        if dropthese[i] == 1: # Ignore the comments that were previously determined to not have full inner-annotator agreement.
            continue
        val = lstag[i]
        text = ls[i]
        writer.writerow([val, text])