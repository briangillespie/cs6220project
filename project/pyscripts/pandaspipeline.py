from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import pandas as pd
import numpy as np
import re
import sys

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


def stem_word_list(list_of_stopped_token):
    stemmed_tokens = []
    for token in list_of_stopped_token:
        stemmed_tokens.append(p_stemmer.stem(token))
    return stemmed_tokens


def flatten(list_to_flatten):
    return [element for sublist in list_to_flatten for element in sublist]


def find_top_20_words(word_count_tuple):
    return sorted(word_count_tuple, key=lambda x: x[1], reverse=True)[:20]


data = pd.read_csv(BGTRAINING,
                   sep='\t',
                   header=0,
                   names=['uid', 'tweetid', 'body', 'date'],
                   index_col=False,
                   parse_dates=[3],
                   infer_datetime_format=True,
                   engine='c',
                   error_bad_lines=False)

# cleantext = data['body'].apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', str(x)).lower())
data = add_new_column(data,
                        data['body'].apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', str(x)).lower()),
                        'tokens')
print(data[0:16])

tokenizer = RegexpTokenizer(r'\w+')
data['tokens'] = data['tokens'].apply(lambda x: tokenizer.tokenize(str(x)))
# result = add_new_column(result,
#                         result['cleantext'].map(lambda x: tokenizer.tokenize(str(x))),
#                         'tokens')


en_stopper = get_stop_words('en')
data['tokens']= data['tokens'].map(lambda x: stop_token_list(x, en_stopper))
# result = add_new_column(result, stopped, 'stopped')
# # print(result[:16])
#
p_stemmer = PorterStemmer()
data['tokens'] = data['tokens'].map(lambda x: stem_word_list(x))
# result = add_new_column(result, stemmed, 'stemmed')
# print(result[:16])

# columns_to_keep = ['uid', 'stemmed']
# result[columns_to_keep].to_csv('outputstemmed.txt', ',')

# word_count = data['body'].map(lambda x: len(x))
# result = add_new_column(result, word_count, 'word_count')
# print(result[:16])

# dictionary = corpora.Dictionary(data['body'])
# vocab_count = len(dictionary)
#
# corpus = [dictionary.doc2bow(flatten(data['body']))]
#
# top_20_words = find_top_20_words(flatten(corpus))
# top_word_dict = {}
# for top_word in top_20_words:
#     top_word_dict[dictionary.get(top_word[0])] = top_word[1]
#
#
# gb = data.groupby(["uid"]).size().reset_index(name='count')
# top_20_users = gb.sort(['count'], ascending=False)[:20]
#
# print "Total number of words", data['word_count'].sum()
# print "No. of words in Vocabulary", vocab_count
# print "Average words in a tweet:", data['word_count'].mean()
# print "No of words  in Longest tweet:", data['word_count'].max()
# min_words = data['word_count'].min()
# print "No of words in shortest tweet:", 1 if  min_words == 0 else min_words #some tweets only contain a link
# print "Top 20 words and their counts:", top_word_dict
# print "Top 20 users and their tweet counts", top_20_users
#
# # ldamodel = models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
# #
# # print(ldamodel.print_topics(num_topics=3, num_words=3))
# # print(ldamodel.print_topics(num_topics=2, num_words=4))