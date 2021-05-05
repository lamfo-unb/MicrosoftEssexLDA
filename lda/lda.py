#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim import models
import pyLDAvis.gensim
import pyLDAvis.sklearn
import os
from gensim.parsing.preprocessing import remove_stopwords
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
import re
from gensim.models.phrases import Phrases
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns



stop_words = stopwords.words('english')

stop_words.extend(["time","week","day","month","cnn","year","going","covid","19","covid-19"])



# In[4]:


df = pd.read_csv("/Users/lucasgomes/Documents/projetos/Essex/MicrosoftEssexScrapers/results/CNN/articles.csv")
df2 = pd.read_csv("/Users/lucasgomes/Documents/projetos/Essex/MicrosoftEssexScrapers/results/WashingtonPost/articles.csv")
df = pd.concat([df,df2])
print("import data")

# In[3]:

df.Text = df.Text.apply(lambda x: str(x))

df.Text = df.Text.apply(lambda x: remove_stopwords(x))
df.Text = df.Text.apply(lambda x: re.sub(r'\W', ' ', x))
df.Text = df.Text.apply(lambda x: re.sub(r' \w ', ' ', x))
df.Text = df.Text.apply(lambda x: x.lower())
df.Text = df.Text.apply(lambda x: x.split())
lemmatizer = WordNetLemmatizer()
df.Text = df.Text.apply(lambda x: [lemmatizer.lemmatize(token) for token in x] )

df.Text = df.Text.apply(lambda x: [w for w in x if not w in stop_words])


phrase_model = Phrases(df.Text, min_count=1, threshold=1)
df.Text = df.Text.apply(lambda x: phrase_model[x] )

df.Text = df.Text.apply(lambda x: [w for w in x if len(w)>1])





common_texts = df.Text.tolist()


# In[ ]:





# In[ ]:


# Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
# Filter out words that occur less than 20 documents, or more than 50% of the documents.
common_dictionary.filter_extremes(no_below=5, no_above=0.5)

common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]


# In[ ]:


# Considering 1-15 topics, as the last is cut off
num_topics = list(range(30)[1:])
num_keywords = 29


LDA_models = {}
LDA_topics = {}
for i in num_topics:
    print("running model: "+str(i))
    LDA_models[i] = models.LdaModel(corpus=common_corpus,
                             id2word=common_dictionary,
                             num_topics=i,
                             update_every=1,
                             chunksize=len(common_corpus),
                             passes=3,
                             alpha='auto',
                             random_state=42,
                             minimum_probability = 0,
                            minimum_phi_value = 0)

    shown_topics = LDA_models[i].show_topics(num_topics=i, 
                                             num_words=num_keywords,
                                             formatted=False)
    LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]

def jaccard_similarity(topic_1, topic_2):
    """
    Derives the Jaccard similarity of two topics

    Jaccard similarity:
    - A statistic used for comparing the similarity and diversity of sample sets
    - J(A,B) = (A ∩ B)/(A ∪ B)
    - Goal is low Jaccard scores for coverage of the diverse elements
    """
    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))
                    
    return float(len(intersection))/float(len(union))

LDA_stability = {}
for i in range(0, len(num_topics)-1):
    print("running stability: " + str(i))
    jaccard_sims = []
    for t1, topic1 in enumerate(LDA_topics[num_topics[i]]): # pylint: disable=unused-variable
        sims = []
        for t2, topic2 in enumerate(LDA_topics[num_topics[i+1]]): # pylint: disable=unused-variable
            sims.append(jaccard_similarity(topic1, topic2))    
        
        jaccard_sims.append(sims)    
    
    LDA_stability[num_topics[i]] = jaccard_sims
    
mean_stabilities = [np.array(LDA_stability[i]).mean() for i in num_topics[:-1]]
print("calculating Coherence Score")
coherences = [CoherenceModel(model=LDA_models[i], corpus=common_corpus, dictionary=common_dictionary, coherence='u_mass').get_coherence()              for i in num_topics[:-1]]
print("Finished calculating Coherence Score")
coh_sta_diffs = [coherences[i] - mean_stabilities[i] for i in range(num_keywords)[:-1]] # limit topic numbers to the number of keywords
coh_sta_max = max(coh_sta_diffs)
coh_sta_max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max]
ideal_topic_num_index = coh_sta_max_idxs[0] # choose less topics in case there's more than one max
ideal_topic_num = num_topics[ideal_topic_num_index]

plt.figure(figsize=(20,10))
ax = sns.lineplot(x=num_topics[:-1], y=mean_stabilities, label='Average Topic Overlap')
ax = sns.lineplot(x=num_topics[:-1], y=coherences, label='Topic Coherence')

ax.axvline(x=ideal_topic_num, label='Ideal Number of Topics', color='black')
ax.axvspan(xmin=ideal_topic_num - 1, xmax=ideal_topic_num + 1, alpha=0.5, facecolor='grey')

y_max = max(max(mean_stabilities), max(coherences)) + (0.10 * max(max(mean_stabilities), max(coherences)))
ax.set_ylim([0, y_max])
ax.set_xlim([1, num_topics[-1]-1])
                
ax.axes.set_title('Model Metrics per Number of Topics', fontsize=25)
ax.set_ylabel('Metric Level', fontsize=20)
ax.set_xlabel('Number of Topics', fontsize=20)
plt.legend(fontsize=20)
plt.show()   

# Train the model on the corpus.
# lda = models.LdaModel(common_corpus, num_topics=10, minimum_probability=0)


# In[ ]:


print(LDA_models[ideal_topic_num_index+1])


# In[ ]:


LDA_models[ideal_topic_num_index+1].save("../../MicrosoftEssexHeroku/ldamodel")
common_dictionary.save("../../MicrosoftEssexHeroku/ldadic")
phrase_model.save("../../MicrosoftEssexHeroku/phaser")


# In[ ]:


p = pyLDAvis.gensim.prepare(LDA_models[ideal_topic_num_index+1], common_corpus, common_dictionary, n_jobs=-1, sort_topics=False)

pyLDAvis.save_html(p, "../../MicrosoftEssexHeroku/FONTE".replace("FONTE","LDA.html"))


# In[ ]:





# In[ ]:


pyLDAvis.display(p, local = True)


# In[ ]:


md = "# Examples for each topic \n"
for i in range(0,len(num_topics)-1):
    md = md  + "\n"
    print(i)
    md = md + "## Topic "+str(i+1) + "\n"
    collected = 0
    for row in df.itertuples():        
        other_corpus = common_dictionary.doc2bow(row.Text)
        vector = LDA_models[ideal_topic_num_index+1][other_corpus]
        topic_percs_sorted = sorted(vector, key=lambda x: (x[1]), reverse=True)
        if topic_percs_sorted[0][0] == i:
            if topic_percs_sorted[0][1] > 0.9:
                md = md +"("+str(collected+1)+") " + row.URL +" "+str(int(topic_percs_sorted[0][1]*100)) + "% \n\n"
                collected += 1
                if collected == 10:
                    break
            if row.Index > 1000:
                if topic_percs_sorted[0][1] > 0.5:
                    md = md +"("+str(collected+1)+") "+ row.URL +" "+str(int(topic_percs_sorted[0][1]*100))+ "% \n\n"
                    collected += 1
                    if collected == 10:
                        break
            if row.Index > 2000:
                if topic_percs_sorted[0][1] > 0.3:
                    md = md  +"("+str(collected+1)+") "+row.URL +" "+ str(int(topic_percs_sorted[0][1]*100)) + "% \n\n"
                    collected += 1
                    if collected == 10:
                        
                        break

print(md)
text_file = open("lda/sites.txt", "w")
n = text_file.write(md)
text_file.close()
                        
        


# In[ ]:




