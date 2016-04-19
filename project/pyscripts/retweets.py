from __future__ import print_function
import gensim
from gensim import corpora, models
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
from stop_words import get_stop_words
from nltk import RegexpTokenizer

DATA = 'trainingData.txt'
ALL_TRAINING = 'training_set_tweets.txt'
dd = 'dd.txt'
OUTPUT = open('out.txt', 'w+')
retweet_token = 'rt'
regex = r'(\s*)@\w+|[^a-zA-Z ]|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
DATAFRAME = open('dataframe.txt', 'w+')
USER_PROFILE = open('user_profile.txt', 'w+')
STOP_WORDS = ['make', 'just', 'u', 'now', 'going', 'video', 'like', 'know', 'get', 'lol', 'can', 'im', 'go', 'new', 'us',
              'rt', 'good', 'like', 'will', 'come', 'one', 'dont', 'today', 'check', 'back', 'see', 'day', 'tonight',
              'cant', 'want', 'got', 'right', 'still', 'need', 'time', 'week', 'great', 'watch', 'looking', 'hour',
              'love', 'free', 'end', 'live', 'think', 'thanks', 'let', 'awesome', 'whats', 'night', 'wait', 'makes',
              'making', 'first', 'last', 'take', 'nan', 'didnt']


def remove_stop_words(row, stop_words_list):
    stopped_tokens = [token for token in row if token not in stop_words_list and STOP_WORDS]
    return stopped_tokens


def stem_tokens(tweet):
    stemmed_tweet = [stemmer.stem(token) for token in tweet]
    return stemmed_tweet


data = pd.read_csv(ALL_TRAINING,
                   sep='\t',
                   header=None,
                   names=['uid', 'tweetid', 'body', 'date'],
                   index_col=False,
                   parse_dates=[3],
                   infer_datetime_format=True,
                   engine='c',
                   error_bad_lines=False)

# only keeping uid and tweets, dropping the rest
data.drop(data.columns[[1, 3]], axis=1, inplace=True)
print(data.head())

# removing links, tags, numbers, punctuations
data['body'] = data['body'].apply(lambda x: re.sub(regex, '', str(x)).lower())

tokenizer = RegexpTokenizer(r'\w+')
data['body'] = data['body'].apply(lambda x: tokenizer.tokenize(str(x)))

# remove all the retweets from the data body
criterion = data['body'].map(lambda x: (retweet_token not in x))
data = data[criterion]

# remove the stop words
stop_words = get_stop_words("en")
data['body'] = data['body'].map(lambda x: (remove_stop_words(x, stop_words)))

# stemming the tweets
stemmer = SnowballStemmer("english")
data['body'] = data['body'].map(lambda x: stem_tokens(x))

# saving this dataframe in a CSV for later use
data.to_csv(DATAFRAME)
data = data.groupby('uid').agg(lambda x: x.sum()).reset_index()
data.to_csv(USER_PROFILE)

dictionary = corpora.Dictionary(data['body'])
corpus = [dictionary.doc2bow(text) for text in data['body']]
# print(corpus)

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

model = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, alpha='auto', num_topics=10, passes=10)
model.save('tweets.lda')

print(model.print_topics(num_topics=10, num_words=3))
