### Import the required libraries:

import pandas as pd
import csv

## This notebook balances a dataset by minority class label.
from google.colab import drive
drive.mount('/content/gdrive')

file_in = "/content/gdrive/My Drive/Colab Notebooks/various_json/localize/new_localize_data_parsed.csv"
data_col = "REVIEW"
label_col = "TAG"
df = pd.read_csv(file_in, engine = 'python');
df.sort_values(by=['UPDT_TIME'], inplace=True, ascending=False) # Sort by timestamp
df = df.reset_index() # remove later
print("Number of samples:", len(df))
df.head(5)

def num_balanced_labels(data_col, label_col, df):
    com_list = []
    num_0 = 0
    num_1 = 0
    print("Balance: Total observations to parse:", len(df))
    for i in range(len(df)):
        if i%2000 == 0:
            print("Update:", i)
        sen = df[data_col][i]
        val = int(df[label_col][i])
        if sen == "" or isinstance(sen, str) == False or sen == " ":
                continue
        sen = sen.strip()
        if sen in com_list:
            continue
        com_list.append(sen)
        if val == 1:
            num_1 = num_1 + 1
        else:
            num_0 = num_0 + 1
    num_labels = min(num_0, num_1)
    com_list = []
    print("The number of observations of class False is:", num_0)
    print("The number of observations of class True is:", num_1)
    return(num_labels)

file_out = "/content/gdrive/My Drive/Colab Notebooks/various_json/localize/new_localize_data_balanced.csv"


### The code below creates the new cleaned dataset by writing it out to a new csv:
num_labels = num_balanced_labels(data_col, label_col, df)
ls = []
num_0 = 0 # Negative class label
num_1 = 0 # Positive class label
with open(file_out, 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow([label_col, data_col])
    for i in range(len(df)):
        if i%2000 == 0:
            print("Update:", i) # Tracks progress
        text = df[data_col][i]
        if text == "" or isinstance(text, str) == False or text == " ":
            continue
        text = text.strip()
        if text in ls:
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
        writer.writerow([val, text])

print("The number of observations in the original dataset is:", len(df))
print("The number of observations in the new dataset is:", len(ls))
print("The number of observations of each class is:", num_labels)


