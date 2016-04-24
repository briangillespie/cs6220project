from scipy.stats import entropy
from numpy.linalg import norm
import gensim
import numpy as np


def jsd(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def word2id(word, dictionary):
    return dictionary.doc2bow([word])[0][0]

if __name__ == '__main__':
    lda_model = '../tweets.lda'
    lda = gensim.models.LdaModel.load(lda_model)
    dictionary = lda.id2word

    topic_dists = [0] * lda.num_topics
    base_dist = [0.0]*len(dictionary)
    new_topic_dist = base_dist

    for i in range(lda.num_topics):
        for word in lda.show_topic(i, topn=len(dictionary)):
            new_topic_dist[word2id(word[0], dictionary)] = word[1]
        topic_dists[i] = sorted(new_topic_dist)
        new_topic_dist = base_dist

    divs = [[0.0 for x in range(lda.num_topics)] for y in range(lda.num_topics)]

    for i in range(lda.num_topics):
        for j in range(lda.num_topics):
            if i != j:
                divs[i][j] = jsd(topic_dists[i], topic_dists[j])

    def spec_min(row, i):
        minimum = 1.1
        index = -1
        for j in range(len(row)):
            if j != i:
                minimum = min(minimum, row[j])
            index = j if minimum == row[j] else index
        return minimum, index


    def spec_max(row):
        maximum = -1.0
        index = -1
        for j in range(len(row)):
            if j != i:
                maximum = max(maximum, row[j])
                index = j if maximum == row[j] else index
        return maximum, index


    for i in range(lda.num_topics):
        print "topic " + str(i) + ". Max: " + str(spec_max(divs[i])) + " and Min: " + str(spec_min(divs[i], i))
