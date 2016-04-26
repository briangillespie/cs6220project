from __future__ import print_function
import gensim
from gensim import corpora, models

OUTPUT = open("C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output/lda_models_tweets.txt", 'w+')
dictionary = corpora.Dictionary.load("tweets_dict.dict")
tfidf_tweets = corpora.MmCorpus.load("C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output/tweets_tfidf.tfidf_model")
tfidf_up = corpora.MmCorpus.load("C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output/up_tfidf.tfidf_model")


def perform_lda(corpus_tfidf, no_of_topics, lda_file):
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
    print("\n", file=OUTPUT)

# get lda models saved
perform_lda(tfidf_tweets, 50, 'C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output/tweets_50.lda')
print("tweets_50.lda created")


















# perform_lda(tfidf_tweets, 100, 'C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output/tweets_100.lda')
# print("tweets_100.lda created")
# perform_lda(tfidf_tweets, 200, 'C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output/tweets_200.lda')
# print("tweets_200.lda created")
#
# perform_lda(tfidf_up, 20, 'C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output/tweets_up_20.lda')
# print("tweets_up_20.lda created")
# perform_lda(tfidf_up, 40, 'C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output/tweets_up_40.lda')
# print("tweets_up_40.lda created")
# perform_lda(tfidf_up, 70, 'C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output/tweets_up_70.lda')
# print("tweets_up_70.lda created")
# perform_lda(tfidf_up, 100, 'C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output/tweets_up_100.lda')
# print("tweets_up_100.lda created")
