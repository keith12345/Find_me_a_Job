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


df = pd.read_pickle('../Data/01_sf_recommender_pca')
df.reset_index(inplace=True, drop=True)

def get_input():
	return input()

def format_text(text):

	text = re.sub('[\W_]+', ' ', text)
	text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text.lower())

	stop = set(stopwords.words('english'))
	text = [item for item in text.split() if item not in stop]

	lem = WordNetLemmatizer()
	text = [lem.lemmatize(token) for token in text]

	stop2 = ['data', 'code', 'team', 'work']
	text = [item for item in text if item not in stop2]

	return text



def create_ngrams(text):

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

	pickle_in = open('Tools_and_models/top_ngrams','rb')
	top_ngrams = pickle.load(pickle_in, errors='loose')
	pickle_in.close()

	top_ngrams = top_ngrams.split()

	for doc_ngram in doc_ngrams:
		for top_ngram in top_ngrams:
			if doc_ngram == top_ngram:
				text.append(doc_ngram)
			else:
				pass
	return ' '.join(text)


def vectorize(text):
	pickle_in = open('Tools_and_models/tf_idf_vectorizer','rb')
	tf_idf_vectorizer = pickle.load(pickle_in, errors='loose')
	pickle_in.close()

	pickle_in = open('Tools_and_models/tf_idf_model','rb')
	tf_idf = pickle.load(pickle_in, errors='loose')
	pickle_in.close()

	text_vector_array = tf_idf_vectorizer.transform([text]).toarray()

	text_vector = pd.DataFrame(text_vector_array,
					columns=tf_idf.get_feature_names())

	return text_vector


def pcaitize(text_vector):

	pickle_in = open('Tools_and_models/pca_tool','rb')
	pca = pickle.load(pickle_in, errors='loose')
	pickle_in.close()

	text_pca = pca.transform(text_vector)

	return pd.DataFrame(text_pca)

def recommender(df, text_pca,n_recommendations=5):
	# find cosine similarity
	recos = (pd.DataFrame(df.iloc[:,4:].T
			.apply(lambda x: np.dot(x,text_pca.iloc[0,:]))))
	largest_indeces = recos[0].nlargest(n_recommendations).index
	recommendations = df.iloc[largest_indeces].posting_url
	return recommendations


app = flask.Flask(__name__)

@app.route("/")
def viz_page():
	with open("templates/job_recommender.html", 'r') as viz_file:
		return viz_file.read()

@app.route("/recommendations", methods=["POST"])
def answer():
	def make_recommendations(text, df, n_recommendations=5):
		formatted_text = format_text(text)
		ngramed_text = create_ngrams(formatted_text)
		text_vector = vectorize(ngramed_text)
		text_pca = pcaitize(text_vector)
		recommendations = recommender(df, text_pca,n_recommendations=5)
		return recommendations.values

	data = flask.request.json
	test_case_list = data["question"]
	tmp = test_case_list[0]
	#Create a dictionary from ‘processed_docs’ containing the number of times a word appears in the training set.

	answer = list(make_recommendations(tmp,df))



	#return flask.jsonify({'answer':answer})
	return flask.render_template('job_recommender.html',recotest='string')



@app.route("/plotly", methods=["GET"])
def plotly():
    return flask.render_template('plotly_embedded.html')

if __name__=="__main__":
	app.run(debug=True, threaded=True)
