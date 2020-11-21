# Make sure you pip installed these 2 libs first
from bs4 import BeautifulSoup
import pandas as pd
import os

cwd = os.getcwd()

def remove_tags(filename):
    os.chdir("../../Datasets")

    # Input data needs to be in xlsx for mat to avoid some encoding issue
    df = pd.read_csv(filename, encoding='ISO-8859-1')
    # The would be the comments column that needs to be cleaned
    print(df['Comments'].head())
    def remove_tag(string_with_tag):
        try:
            soup = BeautifulSoup(string_with_tag, 'html5lib')
            text = soup.get_text()
            text = text.replace("Â","")
        except:
            # Normally this is caused by an empty cell
            print(string_with_tag)
            text = ''
        return text

    df['Comments'] = df['Comments'].apply(remove_tag)
    print(df['Comments'].head())

    os.chdir(cwd)
    df.drop_duplicates(keep=False, inplace=True)
    df.to_csv(filename, encoding = 'utf8')

remove_tags(filename="suggestions_expertiza_fall2018.csv")
remove_tags(filename="problems_expertiza_fall2018.csv")

