from gensim import corpora, models
import pandas as pd
from ast import literal_eval

path = "C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output/"

TWEETS = path + "tweets.txt"
USER_PROFILE = path + "user_profile.txt"
TWEET_CORPUS = path + "tweets_corpora.mm"
USER_PROFILE_CORPUS = path + "tweets_corpora_up.mm"
TWEET_TFIDF = path + "tweets_tfidf.tfidf_model"
USER_PROFILE_TFIDF = path + "up_tfidf.tfidf_model"
TWEET_DICT_NAME = path + "tweets_dict.dict"
USER_PROFILE_DICT_NAME = path + "user_profile_dict.dict"


tweets = pd.read_csv(TWEETS,
                     sep=',',
                     header=0,
                     names=['uid', 'body'],
                     engine='c',
                     quotechar='"',
                     error_bad_lines=False,
                     encoding="utf-8")

user_profile = pd.read_csv(USER_PROFILE,
                           sep=',',
                           header=0,
                           names=['uid', 'body'],
                           engine='c',
                           error_bad_lines=False,
                           encoding="utf-8")

body = tweets['body'].apply(lambda x: literal_eval(x)).values
tweets['body'] = body
body = user_profile['body'].apply(lambda x: literal_eval(x)).values
user_profile['body'] = body


def get_dictionary(df, dict):
    dictionary = corpora.Dictionary(df['body'])
    # remove the words that occur only once in the dictionary
    once_ids = [token_id for token_id, doc_freq in dictionary.dfs.iteritems() if doc_freq == 1]
    dictionary.filter_tokens(once_ids)
    dictionary.save(dict)
    print("dictionary created and saved")
    return dictionary


def create_corpora(tfidf_file, df, dictionary, corpus_file):
    corpus = [dictionary.doc2bow(text) for text in df['body']]
    corpora.MmCorpus.serialize(corpus_file, corpus, id2word=dictionary)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    corpora.MmCorpus.serialize(tfidf_file, corpus_tfidf)


tweet_dict = get_dictionary(tweets, TWEET_DICT_NAME)
user_profile_dict = get_dictionary(user_profile, USER_PROFILE_DICT_NAME)

create_corpora(TWEET_TFIDF, tweets, tweet_dict, TWEET_CORPUS)
create_corpora(USER_PROFILE_TFIDF, user_profile, user_profile_dict, USER_PROFILE_CORPUS)

