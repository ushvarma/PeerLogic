### Import the required libraries:

import pandas as pd
import csv
from preprocessing_utils import num_balanced_labels
from preprocessing_utils import write_balanced_data


file_in = "/content/gdrive/My Drive/Colab Notebooks/various_json/localize/new_localize_data_parsed.csv"
file_out = "outputfile.csv"

data_col = "REVIEW"
label_col = "TAG"
df = pd.read_csv(file_in, engine = 'python');
df.sort_values(by=['UPDT_TIME'], inplace=True, ascending=False) # Sort by timestamp
df = df.reset_index() # remove later
print("Number of samples:", len(df))
df.head(5)

### The code below creates the new cleaned dataset by writing it out to a new csv:
num_labels = num_balanced_labels(data_col, label_col, df)
blanced_dataset = write_balanced_data(data_col, label_col,df,file_out)

print("The number of observations in the original dataset is:", len(df))
print("The number of observations in the new dataset is:", len(blanced_dataset))
print("The number of observations of each class is:", num_labels)


