'''
Author:Fairuz Shadmani Shishir
16.05.2020
insidemaps.com

'''

#import libraries

#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
#from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
from flask_cors import CORS

#import the excel file
#file_location=r'E:\InsideMaps Office\KnowledgeBase.xlsx'

df = pd.read_excel('KnowledgeBase.xlsx')

vectorizer = CountVectorizer()
bow_features = vectorizer.fit_transform(df['Query'])
bow_features.get_shape()

# TFIDF vectorizer
tfidf = TfidfVectorizer()
tfidf_features = tfidf.fit_transform(df.Query)
tfidf_features.get_shape()

#re for reular expressions
import re

def process_query(query):
    preprocessed_reviews = []
    sentance = re.sub("\S*\d\S*", "", query).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    #sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords.words('english'))
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower())
    preprocessed_reviews.append(sentance.strip())
    return preprocessed_reviews
    
def tfidf_search(tfidf, query):
    query = process_query(query)
    query_trans = tfidf.transform(query)
    pairwise_dist = pairwise_distances(tfidf_features, query_trans)
    
    indices = np.argsort(pairwise_dist.flatten())[0:5]
    df_indices = list(df.index[indices])
    return df_indices
    

def bow_search(vectorizer, query):
    query = process_query(query)
    query_trans = vectorizer.transform(query)
    pairwise_dist = pairwise_distances(bow_features, query_trans)
    
    indices = np.argsort(pairwise_dist.flatten())[0:5]
    df_indices = list(df.index[indices])
    return df_indices
    
    
    
def search(query, typ = "tfidf"):
    query_list=[]
    answer_list=[]
    if typ == "tfidf":
        val = tfidf_search(tfidf, query)
    else :
        val = bow_search(vectorizer, query)
        
    for i in (val):   
        query_list.append(df.Query.iloc[i]) 
        answer_list.append(df.Answers.iloc[i])
    return query_list,answer_list
    
    

#query = "insidemaps"
#Queries,Answers= search(query)

from flask import Flask, request, redirect, url_for, flash, jsonify
#import pickle as p
#import json


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def makecalc():
    data = request.get_json()
    Queries,Answers= search("insidemaps")
    #print(Queries[0])
    #print(Answers[0])
    return jsonify({'query':Queries,'answers':Answers})

@app.route('/getAll', methods=['POST'])
def get_all_data():
    data = request.get_json()
    query = data['key']
    Queries, Answers = search(query)
    return jsonify({'query': Queries, 'answers': Answers})


if __name__ == '__main__':
    #modelfile = 'final_prediction.pickle'
    #model = p.load(open(modelfile, 'rb'))
    #app.run(debug=True,port=30, host='127.0.0.1')
    app.run(debug=True)