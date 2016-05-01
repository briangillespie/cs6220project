from scipy.stats import entropy
from numpy.linalg import norm
import gensim
import numpy as np


def jsd(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def hellinger_dist(dense1, dense2):
    return np.sqrt(0.5 * ((np.sqrt(dense1) - np.sqrt(dense2))**2).sum())


def word2id(word, dictionary):
    return dictionary.doc2bow([word])[0][0]


def get_sparse_topic_dist(lda, topicid):
    vocab_size = len(lda.id2word)
    return [(word2id(word[0], lda.id2word), word[1]) for word in lda.show_topic(topicid, topn=vocab_size)]


def sparseTD2DenseTD():
    return None

if __name__ == '__main__':
    lda_model = '../tweets.lda'
    lda = gensim.models.LdaModel.load(lda_model)

    topic_dists = [get_sparse_topic_dist(lda, i) for i in range(lda.num_topics)]

    dense_topic_dists = [gensim.matutils.sparse2full(topic_dist, len(lda.id2word)) for topic_dist in topic_dists]

    diffs = [[x for x in range(lda.num_topics)] for y in range(lda.num_topics)]

    for i in range(lda.num_topics):
        for j in range(lda.num_topics):
            diffs[i][j] = hellinger_dist(dense_topic_dists[i], dense_topic_dists[j])

    print diffs

    maxes = []
    mins = []
    for i in range(lda.num_topics):
        maximum = max(x for x in diffs[i] if x > 0)
        max_index = diffs[i].index(maximum)
        minimum = min(x for x in diffs[i] if x > 0)
        min_index = diffs[i].index(minimum)
        maxes.append((maximum, max_index, i))
        mins.append((minimum, min_index, i))
        print "topic " + str(i) + ". Max: " + str((maximum, max_index)) + " and Min: " + str((minimum, min_index))

    curr = 1.1
    index = None
    for i in range(len(mins)):
        if mins[i][0] < curr:
            index = i
            curr = mins[i][0]
    print mins[index]
    print lda.show_topic(mins[index][1]), lda.show_topic(mins[index][2])

    curr = -1.0
    index = None
    for i in range(len(maxes)):
        if maxes[i][0] > curr:
            index = i
            curr = maxes[i][0]
    print maxes[index]
    print lda.show_topic(maxes[index][1]), lda.show_topic(maxes[index][2])
