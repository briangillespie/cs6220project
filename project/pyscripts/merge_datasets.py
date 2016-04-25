import pandas as pd
import numpy as np

TRAINING = 'training_set_tweets.txt'
TEST = 'test_set_tweets.txt'

TRAINING_OUT = open('training_datav2.txt', 'w+')
TEST_OUT = open('test_datav2.txt', 'w+')

training = pd.read_csv(TRAINING,
                   sep='\t',
                   header=None,
                   names=['uid', 'tweetid', 'body', 'date'],
                   index_col=False,
                   parse_dates=[3],
                   infer_datetime_format=True,
                   engine='c',
                   error_bad_lines=False)

test = pd.read_csv(TEST,
                    sep='\t',
                    header=None,
                    names=['uid', 'tweetid', 'body', 'date'],
                    index_col=False,
                    parse_dates=[3],
                    infer_datetime_format=True,
                    engine='c',
                    error_bad_lines=False)

split = np.random.rand(len(test)) < 0.998
test2training = test[split]
test = test[~split]
print len(test2training), len(test)

training = pd.concat([training, test])

training.to_csv(TRAINING_OUT)
test.to_csv(TEST_OUT)

