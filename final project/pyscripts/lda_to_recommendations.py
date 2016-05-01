from __future__ import print_function
import gensim
import re
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from stop_words import get_stop_words
from nltk import RegexpTokenizer
from random import randint

RECOM_TWEETS = open("result/bow_u_40.txt", "w+")
lda_model = "output\\tweets_up_40_c_bow.lda"
stoplist = get_stop_words("en")
retweet_token = 'rt'
regex = r'(\s*)@\w+|[^a-zA-Z ]|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
stop_words_tweets = ['make', 'just', 'u', 'now', 'going', 'video', 'know', 'get', 'can', 'im', 'go', 'new', 'us',
                 'like', 'will', 'come', 'one', 'dont', 'today', 'check', 'back', 'see', 'day',
                 'cant', 'want', 'got', 'right', 'still', 'need', 'time', 'week', 'watch', 'looking', 'hour',
                 'end', 'let', 'whats', 'makes', 'making', 'first', 'last', 'take', 'nan', 'didnt']
stemmer = SnowballStemmer("english")

stop_words_new = pd.read_csv("data\\all_stop.txt",
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

model = gensim.models.LdaModel.load(lda_model)
dictionary = model.id2word


def pre_process(tweet):
    # removing links, tags, numbers, punctuations
    tweet = re.sub(regex, '', str(tweet)).lower()

    tokenizer = RegexpTokenizer(r'\w+')
    tweet = tokenizer.tokenize(str(tweet))

    stopped_tokens = [token for token in tweet if token not in stop_words_tweets]

    stemmed_tweet = [stemmer.stem(token) for token in stopped_tokens]
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

    # dropping the lines that do not match the required format
    data.dropna(axis=0, how='any', inplace=True)

    # only keeping uid and tweets, dropping the rest
    data.drop(data.columns[[0, 2, 4]], axis=1, inplace=True)

    return data


data = read_data("data\\new_test_data.csv")
print(data[:5])

for tweet in data['body']:
    recommendations_list_3 = list()
    recommendations_list_5 = list()
    recommendations_list_7 = list()
    pre_tweet = pre_process(tweet)
    query = dictionary.doc2bow(pre_tweet)
    topic_vector = model[query]

    sorted_topic_vector = list(sorted(topic_vector, key=lambda x: x[1]))

    first_model = model.show_topic(sorted_topic_vector[-1][0])
    second_model = model.show_topic(sorted_topic_vector[-2][0])

    first_sorted_terms = list(sorted(first_model, key=lambda x: x[1], reverse=True))
    first_recom = first_sorted_terms.pop()
    second_recom = first_sorted_terms.pop()

    second_sorted_terms = list(sorted(second_model, key=lambda x: x[1], reverse=True))
    third_recom = second_sorted_terms.pop()

    recommendations_list_3.append(first_recom[0])
    recommendations_list_3.append(second_recom[0])
    recommendations_list_3.append(third_recom[0])

    recommendations_list_5 = recommendations_list_3[:]
    recommendations_list_7 = recommendations_list_3[:]

    first_recom = first_sorted_terms[randint(0, len(first_sorted_terms) - 1)]
    recommendations_list_5.append(first_recom[0])
    second_recom = second_sorted_terms[randint(0, len(second_sorted_terms) - 1)]
    recommendations_list_5.append(second_recom[0])

    first_recom = first_sorted_terms[randint(0, len(first_sorted_terms) - 1)]
    recommendations_list_7.append(first_recom[0])
    second_recom = first_sorted_terms[randint(0, len(first_sorted_terms) - 1)]
    recommendations_list_7.append(second_recom[0])

    first_recom = second_sorted_terms[randint(0, len(second_sorted_terms) - 1)]
    recommendations_list_7.append(first_recom[0])
    second_recom = second_sorted_terms[randint(0, len(second_sorted_terms) - 1)]
    recommendations_list_7.append(second_recom[0])

    print(tweet, ":", recommendations_list_3, ":", recommendations_list_5, ":", recommendations_list_7, file=RECOM_TWEETS)
