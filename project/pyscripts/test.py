from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import pandas as pd
import numpy as np
import re

BGTRAINING = "C:\Users\Brian\Documents\DataMiningProject\project\data\\training_set_tweets.txt"
BGTEST = "C:\Users\Brian\Documents\DataMiningProject\project\data\\test_set_tweets.txt"
BGSMALLDUMP = "C:\Users\Brian\Documents\GitHub\cs6220project\project\data\smallDump.txt"


def add_new_column(dataframe, series, column_name):
    result = pd.concat([dataframe, series], axis=1)
    result.columns = np.append(dataframe.columns.values, column_name)
    return result


def stop_token_list(list_of_tokens, stop_words):
    stopped_tokens = []
    for token in list_of_tokens:
        if token not in stop_words:
            stopped_tokens.append(token)
    return stopped_tokens


def flatten(list_to_flatten):
    return [element for sublist in list_to_flatten for element in sublist]

data = pd.read_csv(BGSMALLDUMP,
                   sep='\t',
                   header=0,
                   names=['uid', 'tweetid', 'body', 'date'],
                   index_col=False,
                   parse_dates=[3],
                   infer_datetime_format=True,
                   engine='c',
                   error_bad_lines=False)

data['body'] = data['body'].apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', str(x)).lower())

print(data[0:10])
