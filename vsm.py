import pandas as pd
import numpy as np
from math import log
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
from nltk.corpus import stopwords
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess(text):
 tokens = word_tokenize(text)
 tokens = [ps.stem(t) for t in tokens if t.isalpha() and t not in stop_words]
 return tokens

data = pd.read_csv("data.txt")
docs = data["text"].tolist()
query = "information retrieval"
processed_query = preprocess(query)
processed_docs = [preprocess(d) for d in docs]


terms = sorted(set(t for d in processed_docs for t in d))
N = len(docs)
tf_idf= np.zeros((len(terms),N))
for i,term in enumerate(terms):
 df = sum(1 for doc in processed_docs if term in doc)
 idf = log(N/df) if df>0 else 0
 for j,doc in enumerate(processed_docs):
  tf = doc.count(term)
  tf_idf[i,j]= tf*idf


# For Query we need tf idf
tf_idf_query = np.zeros((len(terms)))
for i,term in enumerate(terms):
 tf = processed_query.count(term)
 df = sum(1 for doc in processed_docs if term in doc)
 idf = log(N/df) if df>0 else 0
 tf_idf_query[i]= tf*idf


#to find Similarity we perform Cosine SImilairty so dot product
cosine_similarities =[]
for j in range(N):
 current_doc = tf_idf[j]
 if np.linalg.norm(tf_idf_query)>0 and np.linalg.norm(current_doc):
  cos= np.dot(current_doc,tf_idf_query)/(np.linalg.norm(current_doc)*np.linalg.norm(tf_idf_query))
  cosine_similarities.append((j,cos))
 else:
   cosine_similarities.append((j,0))

print(cosine_similarities)



#to find Similarity we perform Cosine SImilairty so dot product
dice_similarities =[]
for j in range(N):
 current_doc = tf_idf[j]
 if np.linalg.norm(tf_idf_query)>0 and np.linalg.norm(current_doc):
  cos= 2*np.dot(current_doc,tf_idf_query)/(np.linalg.norm(current_doc)**2)+(np.linalg.norm(tf_idf_query)**2)
  dice_similarities.append((j,cos))
 else:
   dice_similarities.append((j,0))

print(dice_similarities)
x =sorted(dice_similarities,key = lambda x:x[1])



#to find Similarity we perform Cosine SImilairty so dot product
jaccard_similarities =[]
for j in range(N):
 current_doc = tf_idf[j]
 if np.linalg.norm(tf_idf_query)>0 and np.linalg.norm(current_doc):
  cos= np.dot(current_doc,tf_idf_query)/(np.linalg.norm(current_doc)**2)+(np.linalg.norm(tf_idf_query)**2)+np.dot(current_doc,tf_idf_query)
  jaccard_similarities.append((j,cos))
 else:
   jaccard_similarities.append((j,0))

print(jaccard_similarities)
jaccard_similarities.sort(key = lambda x:x[1])