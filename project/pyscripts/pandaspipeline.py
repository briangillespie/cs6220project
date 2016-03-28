from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import pandas as pd
import numpy as np
import re

def add_new_column(dataframe, series, columnName):
    result = pd.concat([dataframe, series], axis=1)
    result.columns = np.append(dataframe.columns.values, columnName)
    return result

def stop_token_list(listoftokens, stop_words):
    stopped_tokens = []
    for token in listoftokens:
        if token not in stop_words:
            stopped_tokens.append(token)
    return stopped_tokens

data = pd.read_csv('C:\Users\Brian\Documents\DataMiningProject\project\data\smallDump.txt',
                   sep='\t',
                   header=0,
                   names=['uid', 'tweetid', 'body', 'date'],
                   index_col=False,
                   parse_dates=[3],
                   infer_datetime_format=True,
                   engine='c',
                   error_bad_lines=False)

cleaned = data['body'].apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', str(x)).lower())
result = add_new_column(data, cleaned, 'cleantext')
print(result[10:16])

tokenizer = RegexpTokenizer(r'\w+')
tokenized = result['cleantext'].map(lambda x: tokenizer.tokenize(str(x)))
result = add_new_column(result, tokenized, 'tokens')
print(result[10:16])

en_stopper = get_stop_words('en')
stopped = result['tokens'].map(lambda x: stop_token_list(x, en_stopper))
result = add_new_column(result, stopped, 'stopped')
print(result[:16])

# p_stemmer = PorterStemmer()
#
# for stopped_token in stopped_tokens_list:
#     stemmed_token = [p_stemmer.stem(i) for i in stopped_token]
#     texts.append(stemmed_token)
#
# dictionary = corpora.Dictionary(texts)
#
# corpus = [dictionary.doc2bow(text) for text in texts]
# ldamodel = models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
#
# print(ldamodel.print_topics(num_topics=3, num_words=3))
# print(ldamodel.print_topics(num_topics=2, num_words=4))