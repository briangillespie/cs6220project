from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import re

STOP_WORDS = ['make', 'just', 'u', 'now', 'going', 'video', 'like', 'know', 'get', 'lol', 'can', 'im', 'go', 'new', 'us',
              'rt', 'good', 'like', 'will', 'come', 'one', 'dont', 'today', 'check', 'back', 'see', 'day', 'tonight',
              'cant', 'want', 'got', 'right', 'still', 'need', 'time', 'week', 'great', 'watch', 'looking', 'hour', 'love',
              'free', 'end', 'live', 'think', 'thanks', 'let', 'awesome', 'whats', 'night', 'wait', 'makes', 'making',
              'first', 'last', 'use', 'take', 'nan', 'well', 'say', 'thing', 'click']
STOP_WORDS = [word.decode('utf-8') for word in STOP_WORDS]

WINDOWS = True
SMALLDATA = False

if not WINDOWS:
    DATA = '/home/bng1290/project/data/training_set_tweets.txt'
    OUTPUT = '/home/bng1290/project/data/cleandata.txt'
else:
    DATA = "C:\Users\Brian\Documents\DataMiningProject\project\data\\training_set_tweets.txt"
    OUTPUT = open("C:\Users\Brian\Documents\GitHub\cs6220project\project\data\cleandata.txt", "w+")


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


if SMALLDATA:
    data = pd.read_csv(DATA,
                       sep='\t',
                       names=['userid', 'tweetid', 'tokens', 'date'],
                       index_col=False,
                       parse_dates=[3],
                       infer_datetime_format=True,
                       engine='c',
                       error_bad_lines=False,
                       nrows=1000)
else:
    data = pd.read_csv(DATA,
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

data['tokens'] = data['tokens'].apply(lambda x: re.sub(r'[^a-zA-Z ]|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', str(x)).lower())
print("cleaned")
print(data[:5])

tokenizer = RegexpTokenizer(r'\w+')
data['tokens'] = data['tokens'].apply(lambda x: tokenizer.tokenize(str(x)))
print('tokenized')
print(data[:5])

en_stopper = get_stop_words('en') + STOP_WORDS
print(en_stopper)
data['tokens'] = data['tokens'].apply(lambda x: stop_token_list(x, en_stopper))
print('stopped')
print(data[:5])
#
p_stemmer = PorterStemmer()
data['tokens'] = data['tokens'].apply(lambda x: stem_word_list(x))
print('stemmed')
print(data[:5])

cols = ['userid', 'tokens']
data[cols].to_csv(OUTPUT, ',')
print('saved prepped data')
