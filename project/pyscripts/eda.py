from gensim import corpora, models
import pandas as pd
import numpy as np
from __future__ import print_function

DATA = '/home/bng1290/project/data/trainingdumpprepped.txt'
OUTPUT = '/home/bng1290/project/data/edaOut.txt'

def add_new_column(dataframe, series, column_name):
    result = pd.concat([dataframe, series], axis=1)
    result.columns = np.append(dataframe.columns.values, column_name)
    return result

def flatten(list_to_flatten):
    return [element for sublist in list_to_flatten for element in sublist]

def find_top_20_words(word_count_tuple):
    return sorted(word_count_tuple, key=lambda x: x[1], reverse=True)[:20]

data = pd.read_csv(DATA,
                   sep=',',
				   quotechar='"',
                   header=0,
                   index_col=False,
                   engine='c',
                   error_bad_lines=False)

data = add_new_column(data, data['tokens'].map(lambda x: len(x)), 'word_count')

print("Total number of words" + str(['word_count'].sum()), file=OUTPUT)
print("Average words in a tweet:" + str(data['word_count'].mean()), file=OUTPUT)
print("No of words  in Longest tweet:" + str(data['word_count'].max()), file=OUTPUT)
min_words = data['word_count'].min()
min = 1 if  min_words == 0 else min_words
print("No of words in shortest tweet:" + str(min), file=OUTPUT) #some tweets only contain a link

dictionary = corpora.Dictionary(data['tokens'])
vocab_count = len(dictionary)
print("No. of words in Vocabulary" + str(vocab_count), file=OUTPUT)
corpus = [dictionary.doc2bow(flatten(data['tokens']))]

top_20_words = find_top_20_words(flatten(corpus))
top_word_dict = {}
for top_word in top_20_words:
    top_word_dict[dictionary.get(top_word[0])] = top_word[1]
print("Top 20 words and their counts:" + str(top_word_dict), file=OUTPUT)

data = data.groupby(["uid"]).size().reset_index(name='count')
top_20_users = data.sort(['count'], ascending=False)[:20]
print("Top 20 users and their tweet counts" + str(top_20_users), file=OUTPUT)




