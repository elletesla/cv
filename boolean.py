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

docs = pd.read_csv("data.txt")
docs=docs["text"].to_list()
query="fun and information"
processed_query =preprocess(query)
processed_docs = [preprocess(d) for d in docs]
terms = sorted(set(terms for docs in processed_docs for terms in docs ))
#Create Postings/ inverted docs
postings = defaultdict(list)
for i,doc in enumerate(processed_docs):
 for term in doc:
  if term in terms:
   if i not in postings[term]:
     postings[term].append(i)


# now we must be able to process these queries so

operators =("and","or","not")
def query_processing(query,inverted_docs,num_docs):
 tokens= word_tokenize(query.lower())
 result = None
 operator = None
 all_docs = set(range(num_docs))
 i=0
 while i<len(tokens):
  token = tokens[i]
  if token in operators:
   operator = token
   i+=1
   continue
  term = ps.stem(token)
  current_docs = set(postings.get(term,[]))

  if result is None:
   result = current_docs
  if operator =="not":
   current_docs = all_docs-current_docs
   operator = None
  elif operator =="and":
   result = result.intersection(current_docs)
   operator = None
  elif operator =="or":
   result = result.union(current_docs)
   operator = None
  i+=1
 return result if result is not None else  set()

retrieved =query_processing("information and fun",postings,len(docs))