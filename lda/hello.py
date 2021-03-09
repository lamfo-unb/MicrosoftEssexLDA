import streamlit as st
import pandas as pd
import numpy as np
import os, sys
import plotly.graph_objects as go
from gensim import models
from gensim.corpora.dictionary import Dictionary
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.parsing.preprocessing import remove_stopwords



st.title('News classifier - LAMFO & University of Essex & Microsoft AI for Health')

classificacao = False

lda_model = models.LdaModel.load("ldamodel")
common_dictionary = Dictionary.load("ldadic")

def classificalda(x):
    
    x = remove_stopwords(x)
    x =  x.lower()
    x =  x.split()
    lemmatizer = WordNetLemmatizer()
    x =  [lemmatizer.lemmatize(token) for token in x]


    other_corpus = common_dictionary.doc2bow(x)
    vector = lda_model[other_corpus]
    topic_percs_sorted = sorted(vector, key=lambda x: (x[1]), reverse=True)
    a = topic_percs_sorted[0][0]
    b = topic_percs_sorted[1][0]
    c = topic_percs_sorted[2][0]
    a_ = topic_percs_sorted[0][1]
    b_ = topic_percs_sorted[1][1]
    c_ = topic_percs_sorted[2][1]
    return (a,b,c,a_,b_,c_)

# Load from file
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    clf = pickle.load(file)

teste = st.text_input("Enter text to classify bellow.")


a,b,c,a_,b_,c_ = classificalda(teste)



d = {'lda1': [a], 'lda2': [b],'lda2': [b],'lda3': [c]}
df = pd.DataFrame(data=d)

classificacao = clf.predict(df)








if not classificacao:
    valor = 0
    cor = "red"
else:
    valor = int(classificacao)
    if valor>3:
        cor = "green"
    else:
        cor = "red"
    
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = valor,
    gauge = {"axis": {
        "range": [0,6]
      },
          "bar": {"color" :cor}},
    title = {'text': "Truthiness"},
    domain = {'x': (0,1), 'y': (0,1)}
))


st.plotly_chart(fig, use_container_width=True)


st.write("This text contais topics: A (A_%); B (B_%); C* (C_%). Check bellow the meaning of each topic.".replace("A_",str(int(100*a_))).replace("B_",str(int(100*b_))).replace("C_",str(int(100*c_))).replace("A",str(a+1)).replace("B",str(b+1)).replace("C*",str(c+1)))


f = open("../results/LDA.html" , "r")
data = f.read()

st.components.v1.html(data, width = 1200, height = 800, scrolling = True)



