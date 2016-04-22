from __future__ import print_function
import gensim
from gensim import corpora, models
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
from stop_words import get_stop_words
from nltk import RegexpTokenizer

ALL_TRAINING = 'training_set_tweets.txt'
OUTPUT = open('out.txt', 'a')
retweet_token = 'rt'
regex = r'(\s*)@\w+|[^a-zA-Z ]|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
DATAFRAME = open('dataframe.txt', 'w+')
USER_PROFILE = open('user_profile.txt', 'w+')

STOP_WORDS = ['make', 'just', 'u', 'now', 'going', 'video', 'know', 'get', 'can', 'im', 'go', 'new', 'us',
              'rt', 'like', 'will', 'come', 'one', 'dont', 'today', 'check', 'back', 'see', 'day',
              'cant', 'want', 'got', 'right', 'still', 'need', 'time', 'week', 'watch', 'looking', 'hour',
              'end', 'let', 'whats', 'makes', 'making', 'first', 'last', 'take', 'nan', 'didnt']


def remove_stop_words(row, stop_words_list):
    stopped_tokens = [token for token in row if token not in stop_words_list and token not in STOP_WORDS]
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

# dropping the lines that do not match the required format
data.dropna(axis=0, how='any', inplace=True)
print("NA Dropped")

# only keeping uid and tweets, dropping the rest
data.drop(data.columns[[1, 3]], axis=1, inplace=True)
print("Dropped cols")

# removing any rows where UID is not a valid UID
criterion = data['uid'].map(lambda x: (isinstance(x, basestring) and x.isdigit()))
data = data[criterion]
data.reset_index(drop=True, inplace=True)
print("Removed invalid UID")

# removing links, tags, numbers, punctuations
data['body'] = data['body'].apply(lambda x: re.sub(regex, '', str(x)).lower())
print("Removed links")

tokenizer = RegexpTokenizer(r'\w+')
data['body'] = data['body'].apply(lambda x: tokenizer.tokenize(str(x)))
print("Tokenized")

# remove all the retweets from the data body
criterion = data['body'].map(lambda x: (retweet_token not in x))
data = data[criterion]
data.reset_index(drop=True, inplace=True)
print("retweets removed")

# remove the stop words
stop_words = get_stop_words("en")
data['body'] = data['body'].map(lambda x: (remove_stop_words(x, stop_words)))
print("stop words removed")

# remove the instances where there is are no words in the tweet remaining after pre-processing
criterion = data['body'].map(lambda x: (len(x) != 0))
data = data[criterion]
data.reset_index(drop=True, inplace=True)
print("removed empty retweets")

# stemming the tweets
stemmer = SnowballStemmer("english")
data['body'] = data['body'].map(lambda x: stem_tokens(x))
print("stemmed")

# saving this dataframe in a CSV for later use
data.to_csv(DATAFRAME)
data_user_profile = data.groupby('uid').agg(lambda x: x.sum()).reset_index()
print(data[:10])
print(data_user_profile[:10])
data_user_profile.to_csv(USER_PROFILE)
print("written to files")


def perform_lda(df, no_of_topics, corpora_file, lda_file):
    dictionary = corpora.Dictionary(df['body'])
    # remove the words that occur only once in the dictionary
    once_ids = [token_id for token_id, doc_freq in dictionary.dfs.iteritems() if doc_freq == 1]
    dictionary.filter_tokens(once_ids)
    print("dictionary created")

    # save the dictionary for later use
    corpus = [dictionary.doc2bow(text) for text in df['body']]
    corpora.MmCorpus.serialize(corpora_file, corpus, id2word=dictionary)
    print("corpus created")

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    model = gensim.models.LdaModel(corpus_tfidf, id2word=dictionary, alpha='auto', num_topics=no_of_topics)
    model.save(lda_file)

    print(model.print_topics(num_topics=no_of_topics, num_words=5), file=OUTPUT)

perform_lda(data, 100, 'tweets_corpora.mm', 'tweets.lda')
perform_lda(data_user_profile, 40, 'tweets_corpora_up.mm', 'tweets_up.lda')
