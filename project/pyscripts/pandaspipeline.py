from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import pandas as pd
import numpy as np
import re


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


data = pd.read_csv('C:\Users\shail_000\Documents\GitHub\cs6220project\project\data\smallDump.txt',
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
#print(result[10:16])

tokenizer = RegexpTokenizer(r'\w+')
tokenized = result['cleantext'].map(lambda x: tokenizer.tokenize(str(x)))
result = add_new_column(result, tokenized, 'tokens')
#print(result[10:16])

en_stopper = get_stop_words('en')
stopped = result['tokens'].map(lambda x: stop_token_list(x, en_stopper))
result = add_new_column(result, stopped, 'stopped')
#print(result[:16])

p_stemmer = PorterStemmer()
stemmed = result['stopped'].map(lambda x: stem_word_list(x))
result = add_new_column(result, stemmed, 'stemmed')
#result['stemmed'].to_csv('outputstemmed.txt', ',')
#print(result[:16])


word_count = result['stemmed'].map(lambda x: len(x))
result = add_new_column(result, word_count, 'word_count')
#print(result[:16])

print "Total number of words", result['word_count'].sum()

dictionary = corpora.Dictionary(result['stemmed'])
vocab_count = len(dictionary)
print "no. of words in Vocabulary", vocab_count


corpus = [dictionary.doc2bow(flatten(result['stemmed']))]

top_20_words = find_top_20_words(flatten(corpus))
top_word_dict = {}
for top_word in top_20_words:
    top_word_dict[dictionary.get(top_word[0])] = top_word[1]

print top_word_dict

print "Average words in a tweet:", result['word_count'].mean()
print "No of words  in Longest tweet:", result['word_count'].max()
print "No of words in shortest tweet:", 1 if result['word_count'].min() == 0 else result['word_count'].min() #some tweets only contain a link


gb = result.groupby(["uid"]).size().reset_index(name='count')
top_20_users = gb.sort(['count'], ascending=False)[:20]
print top_20_users


# ldamodel = models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
#
# print(ldamodel.print_topics(num_topics=3, num_words=3))
# print(ldamodel.print_topics(num_topics=2, num_words=4))