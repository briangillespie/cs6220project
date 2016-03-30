from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import re

BGTRAINING = "/home/bng1290/project/data/training_set_tweets.txt"
SMALLDUMP = "C:\Users\Brian\Documents\GitHub\cs6220project\project\data\smallDump.txt"
BGTEST = "C:\Users\Brian\Documents\DataMiningProject\project\data\\test_set_tweets.txt"
BGSMALLDUMP = "/home/bng1290/project/data/smallDump.txt"


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


def stem_word_list(list_of_stopped_token):
    stemmed_tokens = []
    for token in list_of_stopped_token:
        stemmed_tokens.append(p_stemmer.stem(token))
    return stemmed_tokens


def flatten(list_to_flatten):
    return [element for sublist in list_to_flatten for element in sublist]


def find_top_20_words(word_count_tuple):
    return sorted(word_count_tuple, key=lambda x: x[1], reverse=True)[:20]


data = pd.read_csv(SMALLDUMP,
                   sep='\t',
                   names=['userid', 'tweetid', 'tokens', 'date'],
                   index_col=False,
                   parse_dates=[3],
                   infer_datetime_format=True,
                   engine='c',
                   error_bad_lines=False)

data.drop(data.columns[[1,3]], axis=1, inplace=True)
print("Dropping unused columns...")
print(data[:5])
				   
data['tokens'] = data['tokens'].apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', str(x)).lower())
print("cleaned")
print(data[:5])

tokenizer = RegexpTokenizer(r'\w+')
data['tokens'] = data['tokens'].apply(lambda x: tokenizer.tokenize(str(x)))
# data = add_new_column(data, data['body'].apply(lambda x: tokenizer.tokenize(str(x))), 'tokens')
print('tokenized')
print(data[:5])

en_stopper = get_stop_words('en')
data['tokens'] = data['tokens'].apply(lambda x: stop_token_list(x, en_stopper))
print('stopped')
print(data[:5])

p_stemmer = PorterStemmer()
data['tokens'] = data['tokens'].apply(lambda x: stem_word_list(x))
print(data[:5])
print('stemmed')

cols = ['userid', 'tokens']
data[cols].to_csv('../data/trainingdumpprepped.txt', ',')
print('saved prepped data')
