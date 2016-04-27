from __future__ import print_function
from gensim import corpora, models

path = "/home/brian/Desktop/LDA/output/"
print(path + "lda_models_tweets.txt")
OUTPUT = open(path + "lda_models_tweets.txt", 'w+')

inpath = "/home/brian/Desktop/LDA/cor_pores/"
tweet_dictionary = corpora.Dictionary.load(inpath + "tweets_dict.dict")
user_profile_dictionary = corpora.Dictionary.load(inpath + "user_profile_dict.dict")
corpora_tweets = corpora.MmCorpus(inpath + "tweets_corpora.mm")
corpora_up = corpora.MmCorpus(inpath + "tweets_corpora_up.mm")

def perform_lda(corpus, no_of_topics, lda_file, dictionary):
    print('model started')
    lda = models.LdaModel(corpus=corpus,
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
perform_lda(corpora_tweets, 50, path + "tweets_50_c_bow.lda", tweet_dictionary)
print("tweets_50_c_bow.lda created")
perform_lda(corpora_tweets, 100, path + "tweets_100_c_bow.lda", tweet_dictionary)
print("tweets_100_c_bow.lda created")
perform_lda(corpora_tweets, 200, path + "tweets_200_c_bow.lda", tweet_dictionary)
print("tweets_200_c_bow.lda created")

perform_lda(corpora_up, 20, path + "tweets_up_20_c_bow.lda", user_profile_dictionary)
print("tweets_up_20_c_bow.lda created")
perform_lda(corpora_up, 40, path + "tweets_up_40_c_bow.lda", user_profile_dictionary)
print("tweets_up_40_c_bow.lda created")
perform_lda(corpora_up, 70, path + "tweets_up_70_c_bow.lda", user_profile_dictionary)
print("tweets_up_70_c_bow.lda created")
perform_lda(corpora_up, 100, path + "tweets_up_100_c_bow.lda", user_profile_dictionary)
print("tweets_up_100_c_bow.lda created")