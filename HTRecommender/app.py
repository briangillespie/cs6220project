from flask import Flask, render_template, request
import gensim
import re
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from stop_words import get_stop_words
from nltk import RegexpTokenizer

retweet_token = 'rt'
regex = r'(\s*)@\w+|[^a-zA-Z ]|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
stop_words_tweets = ['make', 'just', 'u', 'now', 'going', 'video', 'know', 'get', 'can', 'im', 'go', 'new', 'us',
                 'like', 'will', 'come', 'one', 'dont', 'today', 'check', 'back', 'see', 'day',
                 'cant', 'want', 'got', 'right', 'still', 'need', 'time', 'week', 'watch', 'looking', 'hour',
                 'end', 'let', 'whats', 'makes', 'making', 'first', 'last', 'take', 'nan', 'didnt']
stemmer = SnowballStemmer("english")

stop_words_new = pd.read_csv("all_stop.txt",
                         sep='\n',
                         header=None,
                         names=['stop_word'],
                         engine='c',
                         error_bad_lines=False)
stop_words_tweets = set(stop_words_tweets)
stop_words_new = set(stop_words_new["stop_word"])
stop_words_nltk = set(get_stop_words("en"))
stop_words_tweets.update(stop_words_new)
stop_words_tweets.update(stop_words_nltk)
lda_model = "tweets_up_40_c_bow.lda"
model = gensim.models.LdaModel.load(lda_model)
dictionary = model.id2word

app = Flask(__name__)


def pre_process(tweet):
    # removing links, tags, numbers, punctuations
    tweet = re.sub(regex, '', str(tweet)).lower()
    print("Removed links")

    tokenizer = RegexpTokenizer(r'\w+')
    tweet = tokenizer.tokenize(str(tweet))
    print("Tokenized")

    stopped_tokens = [token for token in tweet if token not in stop_words_tweets]

    stemmed_tweet = [stemmer.stem(token) for token in stopped_tokens]
    return stemmed_tweet


@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        recommendations_list = list()
        tweet = request.form['tweet']
        tweet = pre_process(tweet)
        print tweet
        query = dictionary.doc2bow(tweet)
        topic_vector = model[query]
        sorted_topic_vector = list(sorted(topic_vector, key=lambda x: x[1]))
        first_model = model.show_topic(sorted_topic_vector[-1][0])
        second_model = model.show_topic(sorted_topic_vector[-2][0])
        first_sorted_terms = list(sorted(first_model, key=lambda x: x[1]))
        second_sorted_terms = list(sorted(second_model, key=lambda x: x[1]))
        recommendations_list.append(first_sorted_terms[-1][0])
        if first_sorted_terms[-2][0] == second_sorted_terms[-1][0]:
                recommendations_list.append(second_sorted_terms[-2][0])
        else:
            recommendations_list.append(first_sorted_terms[-2][0])
        recommendations_list.append(second_sorted_terms[-1][0])

        recommendations = recommendations_list
        # Run the code to guess hastags based on a tweet

        return render_template('index.html', recommendations=recommendations)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
