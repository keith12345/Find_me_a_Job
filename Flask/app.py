import flask
import numpy as np
import pandas as pd
import pickle
import itertools
import json
import seaborn as sns
import math
import nltk, string
import re
import random
import nltk, string,re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.corpora import Dictionary

app = flask.Flask(__name__)

############################################################################
########################## Load necessary pickles ##########################
############################################################################

df = pd.read_pickle('../Data/01_sf_recommender_pca_b')
df.reset_index(inplace=True, drop=True)

pickle_in = open('Tools_and_models/top_ngrams','rb')
top_ngrams = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('Tools_and_models/tf_idf_vectorizer','rb')
tf_idf_vectorizer = pickle.load(pickle_in, errors='loose')
pickle_in.close()

pickle_in = open('Tools_and_models/tf_idf_model','rb')
tf_idf = pickle.load(pickle_in, errors='loose')
pickle_in.close()

pickle_in = open('Tools_and_models/pca_tool','rb')
pca = pickle.load(pickle_in, errors='loose')
pickle_in.close()

def format_text(text):
    """Reads a single string. Removes punctuation and makes text lowercase. Removes stopwords. Returns as a list."""

    text = re.sub('[\W_]+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text.lower())

    stop = set(stopwords.words('english'))
    text = [item for item in text.split() if item not in stop]

    lem = WordNetLemmatizer()
    text = [lem.lemmatize(token) for token in text]

    stop2 = ['data', 'code', 'team', 'work']
    text = [item for item in text if item not in stop2]

    return text

def create_ngrams(text, top_ngrams):
    """Takes in formmatted text and top_ngrams as lists.
    Creates ngrams and checks whether any created ngrams match those in created when looking at the documentself.
    Returns text with matching ngrams as a string."""

    doc_ngrams = []

    for i, j in zip(text, text[1:]):
        bigram = '_'.join([i, j])
        doc_ngrams.append(bigram)

    for i, j, k in zip(text, text[1:], text[2:]):
        trigram = '_'.join([i, j, k])
        doc_ngrams.append(trigram)

    for i, j, k, l in zip(text, text[1:],
						  text[2:], text[3:]):
        quadgram = '_'.join([i, j, k, l])
        doc_ngrams.append(quadgram)

    top_ngrams = top_ngrams.split()
    for doc_ngram in doc_ngrams:
        for top_ngram in top_ngrams:
            if doc_ngram == top_ngram:
                text.append(doc_ngram)
            else:
                pass

    return ' '.join(text)

def vectorize(text, tf_idf_vectorizer, tf_idf):
    """Takes in ngramed text as a string, tf_idf_vectorizer, and tf_idfself.
    Vectorizes text and uses tf_idf for feature names in a DataFrame."""

    text_vector_array = tf_idf_vectorizer.transform([text]).toarray()
    text_vector = pd.DataFrame(text_vector_array,
    				columns=tf_idf.get_feature_names())

    return text_vector

def pcaitize(text_vector, pca):
    """Performs a principal compnenent analysis on the newly created text vector and returns it as a DataFrame."""

    text_pca = pca.transform(text_vector)
    return pd.DataFrame(text_pca)

def recommender(df, text_pca,n_recommendations=5):
	# find cosine similarity
    recos = (pd.DataFrame(df.iloc[:,4:].T
			.apply(lambda x: np.dot(x,text_pca.iloc[0,:]))))
    largest_indeces = recos[0].nlargest(n_recommendations).index
    recommendations = df.iloc[largest_indeces][['company_name','job_title','posting_url']]
    return pd.DataFrame(recommendations)

answer = ''

#############################################################################
########################### Page render functions ###########################
#############################################################################

# Link to plotly
@app.route("/plotly", methods=["GET"])
def plotly():
    return flask.render_template('plotly_embedded.html')

# Pulls up and populates the results page
@app.route("/results", methods=["GET"])
def results():
	answer = pd.read_pickle('answer')
	return flask.render_template('main_page.html', answer = answer)

# main_page load
@app.route("/", methods=["POST", "GET"])
def test():
    def make_recommendations(text, df, n_recommendations=5):
        formatted_text = format_text(text)
        ngramed_text = create_ngrams(formatted_text, top_ngrams)
        text_vector = vectorize(ngramed_text, tf_idf_vectorizer, tf_idf)
        text_pca = pcaitize(text_vector, pca)
        recommendations = recommender(df, text_pca,n_recommendations=5)
        return recommendations.values

	# Collections user input
    data = flask.request.json

	# User input will be empty upon page load. Must account for that occurence.
    if data is None:
        answer = pd.DataFrame()

    else:
        tmp = data["question"][0]

        answer = make_recommendations(tmp,df)
        answer = pd.DataFrame(answer)
        answer.to_pickle('answer')

    return flask.render_template('main_page.html', answer = answer)

if __name__=="__main__":
	app.run(debug=True, threaded=True)
