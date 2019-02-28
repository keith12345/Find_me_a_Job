from flask import Flask
from flask import Flask, render_template, flash, request, redirect, url_for
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

from flask_table import Table, Col

import numpy as np
import pandas as pd

import pickle

import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# import warnings
# warnings.filterwarnings("ignore")

# import nltk
# nltk.download('wordnet')

# App Config
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

# Declare table
class ItemTable(Table):
    company = Col('Company')
    title = Col('Title')
    job_url = Col('URL')
class Item(object):
    def __init__(self, rating, title):
        self.company = company
        self.title = title
        self.job_url = job_url

class ReusableForm(Form):
    text = TextField('text:', validators=[validators.required()])


# Tools for recommendations
df = pd.read_pickle('../Data/01_sf_recommender_pca')

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
    n_recommendations += 1
    recos = (pd.DataFrame(df.iloc[:,4:].T
            .apply(lambda x: np.dot(x,text_pca.iloc[0,:]))))
    largest_indeces = recos[0].nlargest(n_recommendations).index
    recommendations = df.iloc[largest_indeces].posting_url
    return recommendations


def make_recommendations(df, n_recommendations=5):
    text = get_input()
    formatted_text = format_text(text)
    ngramed_text = create_ngrams(formatted_text)
    text_vector = vectorize(ngramed_text)
    text_pca = pcaitize(text_vector)
    recommendations = recommender(df, text_pca,n_recommendations=5)
    return print(recommendations.values)

####################################################################


@app.route("/", methods=['GET', 'POST'])
def home():
    form = ReusableForm(request.form)
    text = ''
    print(form.errors)
    if request.method == 'POST' and form.validate():
        text = request.form.get('text')
        return redirect(url_for('output_recommendations'))
    else:
        return render_template('home.html', form=form, text=text)

@app.route('/output_recommendations', methods=["POST", "GET"])
def output_recommendations():
    form = ''
    return render_template('output_recommendations.html', form=form)

if __name__ == "__main__":
    app.run(debug=True)
