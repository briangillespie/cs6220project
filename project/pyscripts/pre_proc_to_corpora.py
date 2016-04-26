from gensim import corpora, models
import pandas as pd
from ast import literal_eval

TWEETS = "C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output\\tweets.txt"
USER_PROFILE = "C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output\user_profile.txt"
TWEET_CORPUS = 'C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output\\tweets_corpora.mm'
USER_PROFILE_CORPUS = 'C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output\\tweets_corpora_up.mm'
TWEET_TFIDF = 'C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output\\tweets_tfidf.tfidf_model'
USER_PROFILE_TFIDF = 'C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output\up_tfidf.tfidf_model'

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

print(tweets['body'][:10])
print(user_profile[:10])
body = tweets['body'].apply(lambda x: literal_eval(x)).values
tweets['body'] = body
body = user_profile['body'].apply(lambda x: literal_eval(x)).values
user_profile['body'] = body

print(tweets[:10])
print(tweets[:10])
dictionary = corpora.Dictionary(tweets['body'])
# remove the words that occur only once in the dictionary
once_ids = [token_id for token_id, doc_freq in dictionary.dfs.iteritems() if doc_freq == 1]
dictionary.filter_tokens(once_ids)
dictionary.save("tweets_dict.dict")
print("dictionary created and saved")


def create_corpora(tfidf_file):
    tfidf = models.TfidfModel(None, dictionary=dictionary)
    tfidf.save(tfidf_file)
    print(tfidf)

create_corpora(TWEET_TFIDF)

create_corpora(USER_PROFILE_TFIDF)

