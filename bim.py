import pandas as pd
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


def evaluation_metrics(relevant,retrieved):
 retrieved = set(retrieved)
 relevant = set(relevant)
 true_positive = len(retrieved & relevant)
 precision = true_positive/len(retrieved) if len(retrieved)>0 else 0
 recall = true_positive/len(relevant) if len(retrieved)>0 else 0
 f1 = 2*precision*recall/(precision+recall) if precision and recall>0 else 0

 return precision,recall,f1 


#Binary Independance Model

# here what we do is we want to find term relevance score
# we want to quanitfy how the documents are relevant with terms
# here documens will be given in this format since we need relevance for phase 2
# text label Ex: Cats are lovely R 

docs = pd.read_csv("data.txt")
text = docs["text"].to_list()
relevant_docs = docs["label"].to_list()
all_docs = [i for i in range(len(docs))]
rele_ids = [i for i in range(len(docs)) if relevant_docs[i]=="R"]


N_d = len(text)
N_r = len(rele_ids)
print(docs)
processed_docs = [preprocess(d) for d in text]
terms= sorted(set(d for doc in processed_docs for d in doc))
r_k=defaultdict(int)
d_k= defaultdict(int)
for term in terms:
 for doc_id,doc in enumerate(processed_docs):
  if term in doc:
   d_k[term]+=1
   if doc_id in rele_ids:
    r_k[term]+=1

# so now for each term we need to find relevance for we have to find nk and rk 

#Phase 1 We consider most of the document as not relevant so
from math import log
#pk = 0.5 by default random chance
#qk = dk/Nd
#RSV = pk*(1-qk)/(qk)*(1-pk)
#trk = log  pk*(1-qk)/(qk)*(1-pk)
RSV = defaultdict(int)
trk = defaultdict(int)
trk_2 = defaultdict(int)

pk=0.5
for term in terms:
 qk = d_k[term]/N_d
 if pk>0 and pk<1 and qk<1 and qk>0:
  rsv = pk*(1-qk)/(qk)*(1-pk)
  trk[term]= log(rsv)
  RSV[term]= rsv


# phase 2

for term in terms:
 pk = r_k[term]/N_r
 qk = d_k[term]-r_k[term]/N_d-N_r
 if pk>0 and pk<1 and qk<1 and qk>0:
  rsv = pk*(1-qk)/(qk)*(1-pk)
  trk_2[term]= log(rsv)
  RSV[term]= rsv


# phase 2 with smooth

for term in terms:
 pk = r_k[term]+0.5/N_r+1
 qk = d_k[term]-r_k[term]+0.5/N_d-N_r+1
 if pk>0 and pk<1 and qk<1 and qk>0:
  rsv = pk*(1-qk)/(qk)*(1-pk)
  trk_2[term]= log(rsv)
  RSV[term]= rsv

SORTED_RSV =sorted(RSV.items(),key=lambda x:x[1],reverse=True)

tf_idf_2= np.zeros((len(terms),N))
for i,term in  enumerate((terms)):
    for j,doc in enumerate(processed_docs):
        tf_idf_2[i,j] = trk_2[term]

query = "information retrieval"
processed_query = preprocess(query)
tf_idf_query= np.zeros((len(terms)))

N= len(processed_docs)
for i, term in enumerate(terms):
    df = sum(1 for doc in processed_docs if term in docs)
    idf = log(N/df) if df>0 else 0
    tf =processed_query.count(term)


