from __future__ import print_function
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
from stop_words import get_stop_words
from nltk import RegexpTokenizer
import numpy as np
import settings as s

path_output = s.path
ALL_TRAINING = s.path_data + "training_datav2.txt"
ALL_TESTING = s.path_data + "test_datav2.txt"
TWEETS = open(path_output + "tweets.txt", "w+")
USER_PROFILE = open(path_output + "user_profile.txt", "w+")
TEST_TWEETS = open(path_output + "test_tweets.txt", "w+")
retweet_token = 'rt'
regex = r'(\s*)@\w+|[^a-zA-Z ]|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
stop_words_tweets = ['make', 'just', 'u', 'now', 'going', 'video', 'know', 'get', 'can', 'im', 'go', 'new', 'us',
                     'like', 'will', 'come', 'one', 'dont', 'today', 'check', 'back', 'see', 'day',
                     'cant', 'want', 'got', 'right', 'still', 'need', 'time', 'week', 'watch', 'looking', 'hour',
                     'end', 'let', 'whats', 'makes', 'making', 'first', 'last', 'take', 'nan', 'didnt']
stemmer = SnowballStemmer("english")

stop_words_new = pd.read_csv(s.path_data + "all_stop.txt",
                             sep='\n',
                             header=None,
                             names=['stop_word'],
                             engine='c',
                             error_bad_lines=False)
stop_words_tweets = set(stop_words_tweets)
stop_words_new = set(stop_words_new["stop_word"])
stop_words_nltk = set(get_stop_words("en"))
stop_words_tweets.update(stop_words_new)
stop_words_tweets.update(stop_words_nltk)


def add_new_column(dataframe, series, column_name):
    result = pd.concat([dataframe, series], axis=1)
    result.columns = np.append(dataframe.columns.values, column_name)
    return result


def remove_stop_words(row):
    stopped_tokens = [token for token in row if token not in stop_words_tweets]
    return stopped_tokens


def stem_tokens(tweet):
    stemmed_tweet = [stemmer.stem(token) for token in tweet]
    return stemmed_tweet


def read_data(file):
    data = pd.read_csv(file,
                       sep=',',
                       header=None,
                       names=['id', 'uid', 'tweetid', 'body', 'date'],
                       index_col=False,
                       parse_dates=[3],
                       infer_datetime_format=True,
                       engine='c',
                       error_bad_lines=False)
    return data


def pre_process(data):
    # dropping the lines that do not match the required format
    data.dropna(axis=0, how='any', inplace=True)
    print("NA Dropped")

    # only keeping uid and tweets, dropping the rest
    data.drop(data.columns[[0, 2, 4]], axis=1, inplace=True)
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
    data['body'] = data['body'].map(lambda x: (remove_stop_words(x)))
    print("stop words removed")

    # remove the instances where there is are no words in the tweet remaining after pre-processing
    criterion = data['body'].map(lambda x: (len(x) != 0 or len(x) != 1))
    data = data[criterion]
    data.reset_index(drop=True, inplace=True)
    print("removed empty retweets")

    # stemming the tweets
    data['body'] = data['body'].map(lambda x: stem_tokens(x))
    print("stemmed")

    return data


def save_df_to_csv(data, file_name):
    # saving this dataframe in a CSV for later use
    data.to_csv(file_name)
    print("written to files")


data_training = read_data(ALL_TRAINING)
print("No. of tweets in Tweets file before pre-processing = ", len(data_training['body']))
data_user_profile = data_training.groupby('uid').agg(lambda x: x.sum()).reset_index()

data_testing = read_data(ALL_TESTING)
data_testing = pre_process(data_testing)

save_df_to_csv(data_testing, TWEETS)
save_df_to_csv(data_user_profile, USER_PROFILE)
save_df_to_csv(data_testing, TEST_TWEETS)
print("No. of tweets in testing file after pre-processing = ", len(data_testing['body']))
print("No. of tweets in User Profile file after pre-processing= ", len(data_user_profile['body']))
