### The code below automatically formats a dataset to be ready for use by BERT:
import pandas as pd
import csv
import string
from preprocessing_utils import num_balanced_labels

data_col = "your_data"
label_col = "data_label"

### Load in the file to be converted into bert datasets
## Enter the name of the original dataset
BERT_file_in = "original_data.csv"
## Read in the original dataset
# NOTE: The read_csv funciton may not be able to read data with encoding 'utf-8' if the data
#       contains improper values for this encoding. These values must be removed before this
#       step since BERT requires proper data.
df = pd.read_csv(BERT_file_in, engine = 'python', encoding='utf-8');

### Create the three required train/test/dev datasets for bert:
## Creates the three required dataset for BERT using balanced and clean data
## Initialize the lists of text data and the associated labels
dev_coms = []
test_coms = []
train_coms = []
dev_lab = []
test_lab = []
train_lab = []
## Initialize the counts of classes true/false for each dataset
d_no = 0
te_no = 0
tr_no = 0
d_yes = 0
te_yes = 0
tr_yes = 0
## Clean the data and determine/alot the balanced class quantities to each dataset
num_labels = num_balanced_labels(data_col, label_col, df)
split_value = num_labels * 0.1 # This results in 80-10-10% train/test/dev split with balanced data for each dataset
## Portion the data into each list
for i in range(len(df)):
    val = int(df['ID'][i])
    text = df['Label'][i]
    if text == "" or isinstance(text, str) == False or text == " ":
            continue
    text = "".join(filter(lambda char: char in string.printable, text))
    text = text.strip()
    if val == 0:
        if d_no != split_value:
            d_no = d_no + 1
            dev_lab.append(val)
            dev_coms.append(text)
        elif te_no != split_value:
            te_no = te_no + 1
            test_lab.append(val)
            test_coms.append(text)
        else:
            tr_no = tr_no + 1
            train_lab.append(val)
            train_coms.append(text)
    else:
        if d_yes != split_value:
            d_yes = d_yes + 1
            dev_lab.append(val)
            dev_coms.append(text)
        elif te_yes != split_value:
            te_yes = te_yes + 1
            test_lab.append(val)
            test_coms.append(text)
        else:
            tr_yes = tr_yes + 1
            train_lab.append(val)
            train_coms.append(text)
print("The number of observations for the train/test/dev datasets is:")
print(len(train_coms), len(test_coms), len(dev_coms))
## Create the datasets, only the test dataset is supposed to have a header
with open('dev.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    # Unoffical header is ["ID", "Label", "Throwaway", "Text"]
    for i in range(len(dev_coms)):
        val = dev_lab[i]
        text = dev_coms[i]
        writer.writerow([ i, val, 'a', text])
with open('test.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "Label", "sentence"])
    for i in range(len(test_coms)):
        val = test_lab[i]
        text = test_coms[i]
        writer.writerow([ i, val, text])
with open('train.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    # Unoffical header is ["ID", "Label", "Throwaway", "Text"]
    for i in range(len(train_coms)):
        val = train_lab[i]
        text = train_coms[i]
        writer.writerow([ i, val, 'a', text])
## BERT requires tsv files so this creates tsv versions of the newly formed datasets
df = pd.read_csv('dev.csv')
df.to_csv('dev.tsv', sep='\t', index=False, header=False)
df = pd.read_csv('test.csv')
df.to_csv('test.tsv', sep='\t', index=False, header=True)
df = pd.read_csv('train.csv')
df.to_csv('train.tsv', sep='\t', index=False, header=False)