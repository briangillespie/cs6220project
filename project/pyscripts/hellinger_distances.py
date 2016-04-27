from __future__ import print_function
from scipy.stats import entropy
from numpy.linalg import norm
import gensim
import numpy as np

PATH = '/home/brian/Desktop/LDA/output/'
OUTPUT = open(PATH + 'HD/hellinger_out_tup_50_40.txt', 'w+')


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

def distributions4topics(lda):
    topic_dists = [get_sparse_topic_dist(lda, i) for i in range(lda.num_topics)]
    return [gensim.matutils.sparse2full(topic_dist, len(lda.id2word)) for topic_dist in topic_dists]

def get_matrix_of_hdists(P, Q):
    hdists = [[-1]*len(P) for y in range(len(Q))]

    for i in range(len(Q)):
        for j in range(len(P)):
            if (i <= j):
                hdists[i][j] = hellinger_dist(P[j], Q[i])

    return hdists

def get_maxes_and_mins(hdists):
    maxes = []
    mins = []
    Qtopic = 0
    for row in hdists:
        maximum = max(row)
        max_index = row.index(maximum)
        if len([x for x in row if x >= 0]) > 0:
            minimum = min(x for x in row if x >= 0)
            min_index = row.index(minimum)
        else:
            minimum = None
            min_index = 0
        maxes.append({'max':maximum, 'topic_p':max_index, 'topic_q':Qtopic})
        mins.append({'min':minimum, 'topic_p':min_index, 'topic_q':Qtopic})
        Qtopic += 1

    return {'mins':mins, 'maxes':maxes}

def closest_topics(mins):
    curr = 1.1
    result = None
    for minimum in mins:
        if minimum['min'] < curr:
            result = minimum
            curr = minimum['min']
    return result

def farthest_topics(maxes):
    curr = -1.0
    result = None
    for maximum in maxes:
        if maximum['max'] > curr:
            result = maximum
            curr = maximum['max']
    return result

def run(path_P, path_Q, path_Out):
    lda_modelA = PATH + path_P
    lda_modelB = PATH + path_Q
    output = open(PATH +  path_Out, 'w+')
    print("Computing Hellinger Distances for " + path_P + " and " + path_Q + ". Output to " + path_Out)
    print(("Computing Hellinger Distances for " + path_P + " and " + path_Q + ". Output to " + path_Out), file=output)
    print('\n\n', file=output)


    ldaA = gensim.models.LdaModel.load(lda_modelA)
    ldaB = gensim.models.LdaModel.load(lda_modelB)
    print("Models loaded")

    topic_distsP = distributions4topics(ldaA)
    topic_distsQ = distributions4topics(ldaB)
    print("Word distributions generated for each topic")

    hdists = get_matrix_of_hdists(topic_distsP, topic_distsQ)
    print('HDists computed')
    print('Hdists\n', file=output)
    print(str(hdists), file=output)
    print('\n\n', file=output)

    mm = get_maxes_and_mins(hdists)
    print('Max/Mins computed')
    print('Max/Mins\n', file=output)
    print(str(mm), file=output)
    print('\n\n', file=output)

    print("Farthest topics, " + str(farthest_topics(mm['maxes'])), file=output)
    print("Closest topics, " + str(closest_topics(mm['mins'])), file=output)


if __name__ == '__main__':

    run('tweets_50_c_bow.lda','tweets_100_c_bow.lda','HD/hellinger_out_tup_50_40.txt')
