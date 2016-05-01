path = "C:\Users\shail_000\PycharmProjects\TopicModelingLDA\output/"
path_data = "C:\Users\shail_000\PycharmProjects\TopicModelingLDA\data/"
TWEETS_FILE = path + "test_tweets.txt"
t_lda_50_tfidf = path + "tweets_50.lda"
t_lda_100_tfidf = path + "tweets_100.lda"
t_lda_200_tfidf = path + "tweets_200.lda"
u_lda_20_tfidf = path + "tweets_up_20.lda"
u_lda_40_tfidf = path + "tweets_up_40.lda"
u_lda_70_tfidf = path + "tweets_up_70.lda"
u_lda_100_tfidf = path + "tweets_up_100.lda"
t_lda_50_bow = path + "tweets_50_c_bow.lda"
t_lda_100_bow = path + "tweets_100_c_bow.lda"
t_lda_200_bow = path + "tweets_200_c_bow.lda"
u_lda_20_bow = path + "tweets_up_20_c_bow.lda"
u_lda_40_bow = path + "tweets_up_40_c_bow.lda"
u_lda_70_bow = path + "tweets_up_70_c_bow.lda"
u_lda_100_bow = path + "tweets_up_100_c_bow.lda"

TWEET_DICT = path + "tweets_dict.dict"
USER_PROFILE_DICT = path + "user_profile_dict.dict"

lda_t_bow_list = list()
lda_u_bow_list = list()
lda_t_tfidf_list = list()
lda_u_tfidf_list = list()
lda_t_tfidf_list.append(t_lda_50_tfidf)
lda_t_tfidf_list.append(t_lda_100_tfidf)
lda_t_tfidf_list.append(t_lda_200_tfidf)
lda_u_tfidf_list.append(u_lda_20_tfidf)
lda_u_tfidf_list.append(u_lda_40_tfidf)
lda_u_tfidf_list.append(u_lda_70_tfidf)
lda_u_tfidf_list.append(u_lda_100_tfidf)
lda_t_bow_list.append(t_lda_50_bow)
lda_t_bow_list.append(t_lda_100_bow)
lda_t_bow_list.append(t_lda_200_bow)
lda_u_bow_list.append(u_lda_20_bow)
lda_u_bow_list.append(u_lda_40_bow)
lda_u_bow_list.append(u_lda_70_bow)
lda_u_bow_list.append(u_lda_100_bow)