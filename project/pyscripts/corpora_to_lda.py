from __future__ import print_function
import gensim
from gensim import corpora, models

path = "C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output/"
print(path + "lda_models_tweets.txt")
OUTPUT = open(path + "lda_models_tweets.txt", 'w+')
tweet_dictionary = corpora.Dictionary.load(path + "tweets_dict.dict")
user_profile_dictionary = corpora.Dictionary.load(path + "user_profile_dict.dict")
tfidf_tweets = corpora.MmCorpus(path + "tweets_tfidf.tfidf_model")
tfidf_up = corpora.MmCorpus(path + "up_tfidf.tfidf_model")


def perform_lda(corpus_tfidf, no_of_topics, lda_file, dictionary):
    print('model started')
    lda = models.LdaModel(corpus=corpus_tfidf,
                          id2word=dictionary,
                          alpha='auto',
                          num_topics=no_of_topics)
    print("model created")
    # save the lda model

    lda.save(lda_file)
    # write the models to a file
    print(lda.print_topics(num_topics=no_of_topics, num_words=5), file=OUTPUT)

    print(lda.show_topics(num_topics=5, num_words=5))
    print("\n", file=OUTPUT)

# get lda models saved
perform_lda(tfidf_tweets, 50, path + "tweets_50.lda", tweet_dictionary)
print("tweets_50.lda created")
perform_lda(tfidf_tweets, 100, path + "tweets_100.lda", tweet_dictionary)
print("tweets_100.lda created")
perform_lda(tfidf_tweets, 200, path + "tweets_200.lda", tweet_dictionary)
print("tweets_200.lda created")

perform_lda(tfidf_up, 20, path + "tweets_up_20.lda", user_profile_dictionary)
print("tweets_up_20.lda created")
perform_lda(tfidf_up, 40, path + "tweets_up_40.lda", user_profile_dictionary)
print("tweets_up_40.lda created")
perform_lda(tfidf_up, 70, path + "tweets_up_70.lda", user_profile_dictionary)
print("tweets_up_70.lda created")
perform_lda(tfidf_up, 100, path + "tweets_up_100.lda", user_profile_dictionary)
print("tweets_up_100.lda created")
