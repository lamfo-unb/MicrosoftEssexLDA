import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from gensim import models
from gensim.corpora.dictionary import Dictionary
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.phrases import Phrases
import re
from nltk.corpus import stopwords
import nltk
import urllib.request, urllib.error, urllib.parse
import ssl
import requests
nltk.download('stopwords')
nltk.download('wordnet')
st.set_page_config(layout='wide')

api = st.secrets["api"]

f = open("results/LDA.html" , "r")
data = f.read()

st.title('News classifier - LAMFO & University of Essex & Microsoft AI for Health')


# @st.cache
def importdados():
        
    stop_words = set(stopwords.words('english')) 
    # classificacao = False
    lda_model = models.LdaModel.load("lda/ldamodel")
    common_dictionary = Dictionary.load("lda/ldadic")
    phrase_model = Phrases.load("lda/phaser")

    return stop_words, lda_model, common_dictionary, phrase_model

stop_words, lda_model, common_dictionary, phrase_model = importdados()


def classificalda(x):
    x = remove_stopwords(x)
    x =  re.sub(r'\W', ' ', x)
    x =  re.sub(r' \w ', ' ', x)

    x =  x.lower()
    x =  x.split()
    lemmatizer = WordNetLemmatizer()
    x =  [lemmatizer.lemmatize(token) for token in x]
    x = [w for w in x if not w in stop_words]
    x =  phrase_model[x]



    other_corpus = common_dictionary.doc2bow(x)
    vector = lda_model[other_corpus]
    topic_percs_sorted = sorted(vector, key=lambda x: (x[1]), reverse=True)
    a = topic_percs_sorted[0][0]
    b = topic_percs_sorted[1][0]
    c = topic_percs_sorted[2][0]
    a_ = topic_percs_sorted[0][1]
    b_ = topic_percs_sorted[1][1]
    c_ = topic_percs_sorted[2][1]

    if a_ < 0.3:
        st.warning('Results may have low confidence (<30% homogeneity), consider changing or rewriting the input.')
        
    if a_ < 0.2:
        st.error('Results have low confidence, consider changing or rewriting the input.')
        st.stop()
    return (a,b,c,a_,b_,c_)

def factcheck(texto):
    query = "https://factchecktools.googleapis.com/v1alpha1/claims:search?query=PESQUISA&key=API"
    query = query.replace("PESQUISA",texto).replace(" ","%20").replace("API",api)
    query = requests.get(query, timeout=2.50)
    try:
        json = query.json()["claims"]
        print(list(json[0]))
        for i in range(len(json)):
            texto = json[i]["text"]
            try:
                texto +="- Information Author: " + json[i]["claimant"]
            except:
                print(list(json[i]))
            
            texto +=". Reviewer: " + json[i]["claimReview"][0].get("publisher").get("name")
            texto +=" " + json[i]["claimReview"][0].get("url")
            texto +=". Verdict: " + json[i]["claimReview"][0].get("textualRating")
            rating = json[i]["claimReview"][0].get("textualRating")
            if "true" in rating.lower() or "pants on fire" in rating.lower():
                st.success(texto)
            elif "false" in rating.lower():
                st.error(texto)
            else:
                st.info(texto)
    except:
        st.error("Could not find anything ... try writing in a different form")
    

teste = st.text_area("Enter text to classify",height = 200,max_chars = 10000)
url = st.text_input('Or enter URL to be checked')

# url = st.text_area("Or enter URL to be checked",height = 200,max_chars = 1000)


if url and teste:
    st.error('Please fill only one field (URL or text)')
    st.stop()


if not st.button('Process'):
    with st.beta_expander('See LDA topics distribution'):
        st.components.v1.html(data, width = 1200, height = 900, scrolling = True)
    st.stop()

if url:
    tipoinput = "URL"
    if "www" not in url and "http://" not in url and "https://" not in url:
        url = "http://www." + url
    if "www" in url and "http://" not in url and "http://" not in url:
        url = url.replace("www.","http://www.")
    if "www" not in url and "http" in url:
        url = re.sub(r'.*//', 'http://www.', url)    
    print(url)
    try:    
        context = ssl._create_unverified_context()
        response = urllib.request.urlopen(url,context=context)
        webContent = response.read()
        if len(webContent) > 10000:
            teste = webContent[0:10000]
        teste = webContent
    except Exception as e:
        print(e)
        st.error('URL not valid')
        st.stop()
    
else:
    tipoinput = "given text"
    if len(teste) < 10:   
        st.error('Text provided is too short. Please provide a text with 10 characters or more')
        st.stop()
    with st.beta_expander('Fact checker'):
        factcheck(teste)

    
        

with st.spinner(text='In progress'):
     a,b,c,a_,b_,c_ = classificalda(teste)
     st.success('Done! ' +"Results for "+ tipoinput +". This text contais topics: A (A_%); B (B_%); C* (C_%). Check bellow the meaning of each topic.".replace("A_",str(int(100*a_))).replace("B_",str(int(100*b_))).replace("C_",str(int(100*c_))).replace("A",str(a+1)).replace("B",str(b+1)).replace("C*",str(c+1)))




d = {'lda1': [a], 'lda2': [b],'lda2': [b],'lda3': [c]}
df = pd.DataFrame(data=d)




# st.write()

with st.beta_expander('See LDA topics distribution'):
        st.components.v1.html(data, width = 1200, height = 900, scrolling = True)

with st.beta_expander('See examples for each topic'):

    st.warning("Examples will need to be better choosen, consider them only as one extra reference.")
    f = open("lda/sites.txt" , "r")
    data = f.read()
    st.markdown(data)




for x, x_ in zip([a,b,c],[a_,b_,c_]):
    if x == 0:
        valor = x_ * 100
        
    else:
        valor = 0
    cor = "red"
    
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = valor,
    gauge = {"axis": {
        "range": [0,100]
      },
          "bar": {"color" :cor}},
    title = {'text': "Misinformation"},
    domain = {'x': (0,1), 'y': (0,1)}
))

with st.beta_expander('See percentage of misinformation (Topic 1)'):
    st.plotly_chart(fig, use_container_width=True)


