import pandas as pd
import numpy as np
import re

import json
import gzip

import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
# nltk.download('stopwords')
from nltk.corpus import stopwords
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


# --------------- dataset functions -----------------

# function to parse the dataset in format '*.gz' into a generator json object
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)


# function to format missing value of the dataset
def format_missing_values(df):
    return (df.astype(str)).replace({'[]': pd.NA, '': pd.NA})


# ---------- text preprocessing functions ----------

# function to prepare the features used by the model (concatenation of title and description fields)
def features_selection(df):
    final_features = pd.DataFrame()
    final_features['feature'] = df['title'].str.cat(df['description'], sep=' ')
    return final_features


# function to preprocess the feature (remove punctuation, stopwords etc)
def preprocessing_text(feature, lemmatization = True, stemming = True):
    feature = feature.lower()
    tokens = word_tokenize(feature)
    # remove non-alphabetic characters
    tokens = [token for token in tokens if token.isalpha()]
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # stemming
    if stemming:
        ps = nltk.stem.porter.PorterStemmer()
        tokens = [ps.stem(word) for word in tokens]
    # lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)
