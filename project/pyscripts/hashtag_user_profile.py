import pandas as pd

USER_PROFILE = open('user_profile.txt', 'w+')
INPUT_DATAFRAME = open('dataframe.txt', 'r+')

data = pd.read_csv(INPUT_DATAFRAME,
                   sep=',',
                   header=0,
                   names=['uid', 'body'],
                   engine='c',
                   error_bad_lines=False,
                   encoding="utf-8")


# creating a user profile by grouping the all the tweets from a userID
# such that all tweets of a user are clubbed in one document
data = data.groupby('uid').agg(lambda x: x.sum()).reset_index()
data.to_csv(USER_PROFILE)
