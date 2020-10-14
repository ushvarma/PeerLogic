### This fiel provides code for creating a new dataset that is cleaned of missing values,
### removes duplicate data observations, and balances the data in terms of label proportion.
import pandas as pd
import csv
from bs4 import BeautifulSoup


def clean_data(df):
    return df.dropna()

def remove_html_tag(string_with_tag):
    try:
        soup = BeautifulSoup(string_with_tag, 'html5lib')
        text = soup.get_text()
    except:
        # Normally this is caused by an empty cell
        print(string_with_tag)
        text = ''
    return text

### Function for calculating the smaller proportion of the True/False classes in cleaned data:
## This function determines the count of each class type for future reference in writing a balanced file
def num_balanced_labels(data_col, label_col, df):
    com_list = []
    num_0 = 0
    num_1 = 0
    for i in range(len(df)):
        sen = df[data_col][i]
        val = int(df[label_col][i])
        if sen == "" or isinstance(sen, str) == False or sen == " ":
                continue
        sen = "".join(filter(lambda char: char in string.printable, sen))
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

def write_balanced_data(data_col, label_col,df,file_out):    
    ls = []
    num_labels = num_balanced_labels(data_col, label_col, df)
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
        return ls


