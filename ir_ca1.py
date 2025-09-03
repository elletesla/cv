# -*- coding: utf-8 -*-


#Boolean Model without file import
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Download required NLTK data (run once if needed)
nltk.download("punkt")
nltk.download("punkt_tab")

nltk.download("stopwords")

# Initialize PorterStemmer, stopwords, and operators
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
operators = {"and", "or", "not"}  # Use lowercase operators to match query tokens

# Preprocess function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(t) for t in tokens if t.isalpha() and t not in stop_words and t not in operators]
    return tokens

# Evaluation metrics
def evaluate_query(retrieved, relevant):
    retrieved = set(retrieved)
    relevant = set(relevant)
    true_positives = len(retrieved & relevant)

    # Precision: |Retrieved ∩ Relevant| / |Retrieved|
    precision = true_positives / len(retrieved) if retrieved else 0.0

    # Recall: |Retrieved ∩ Relevant| / |Relevant|
    recall = true_positives / len(relevant) if relevant else 0.0

    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

# Main processing function
def main(sample_data, query):
    # Convert sample data to DataFrame
    df = pd.DataFrame(sample_data, columns=['text', 'label'])
    docs = df['text'].tolist()
    labels = df['label'].tolist()

    # Get relevant document IDs based on labels
    relevant_docs = {i for i, label in enumerate(labels) if label == 'R'}

    # Preprocess documents and query
    preprocess_docs = [preprocess(d) for d in docs]
    preprocess_query = preprocess(query)  # For display only

    # Build inverted index
    terms = sorted(set(t for doc in preprocess_docs for t in doc))
    inverted_index = defaultdict(list)
    for term in terms:
        for doc_id, doc in enumerate(preprocess_docs):
            if term in doc:
                inverted_index[term].append(doc_id)

    # Query processing function
    def query_processing(query, inverted_index, num_docs):
        tokens = word_tokenize(query.lower())  # Use word_tokenize for proper splitting
        result = None
        operator = None
        all_docs = set(range(num_docs))  # Set of all document IDs
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in operators:
                operator = token
                i += 1
                continue
            term = ps.stem(token)
            current_docs = set(inverted_index.get(term, []))
            if operator == "not":
                current_docs = all_docs - current_docs
                operator = None
            if result is None:
                result = current_docs
            elif operator == "or":
                result = result.union(current_docs)
                operator = None
            elif operator == "and":
                result = result.intersection(current_docs)
                operator = None
            i += 1
        return result if result is not None else set()

    # Process query
    result = query_processing(query, inverted_index, len(docs))

    # Evaluate results
    metrics = evaluate_query(result, relevant_docs)

    # Print results
    print("Documents:", docs)
    print("Labels:", labels)
    print("Preprocessed Documents:", preprocess_docs)
    print("Preprocessed Query (terms only):", preprocess_query)
    print("Inverted Index:", dict(inverted_index))
    print("Query:", query)
    print("Relevant Document IDs (from labels):", relevant_docs)
    print("Retrieved Document IDs:", result)
    print("Retrieved Documents:", [docs[i] for i in result])
    print("Evaluation Metrics:", metrics)

# Example usage
if __name__ == "__main__":
    # Sample data
    sample_data = [
        ("Information retrieval is fun.", "R"),
        ("Retrieval of information.", "R"),
        ("Fun with information.", "NR"),
        ("Fun is always fun.", "NR")
    ]

    query = "information"
    main(sample_data, query)

#Boolean Model with file import
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Download required NLTK data (run once if needed)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Initialize PorterStemmer, stopwords, and operators
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
operators = {"and", "or", "not"}  # Use lowercase operators to match query tokens

# Preprocess function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(t) for t in tokens if t.isalpha() and t not in stop_words and t not in operators]
    return tokens

# Read documents from a text file
def read_documents(file_path):
    # Read CSV file with text and label columns, no header
    df = pd.read_csv(file_path, header=None, names=['text', 'label'])
    return df

# Evaluation metrics
def evaluate_query(retrieved, relevant):
    retrieved = set(retrieved)
    relevant = set(relevant)
    true_positives = len(retrieved & relevant)

    # Precision: |Retrieved ∩ Relevant| / |Retrieved|
    precision = true_positives / len(retrieved) if retrieved else 0.0

    # Recall: |Retrieved ∩ Relevant| / |Relevant|
    recall = true_positives / len(relevant) if relevant else 0.0

    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

# Main processing function
def main(file_path, query):
    # Read documents
    df = read_documents(file_path)
    docs = df['text'].tolist()
    labels = df['label'].tolist()

    # Get relevant document IDs based on labels
    relevant_docs = {i for i, label in enumerate(labels) if label == 'R'}

    # Preprocess documents and query
    preprocess_docs = [preprocess(d) for d in docs]
    preprocess_query = preprocess(query)  # For display only

    # Build inverted index
    terms = sorted(set(t for doc in preprocess_docs for t in doc))
    inverted_index = defaultdict(list)
    for term in terms:
        for doc_id, doc in enumerate(preprocess_docs):
            if term in doc:
                inverted_index[term].append(doc_id)

    # Query processing function
    def query_processing(query, inverted_index, num_docs):
        tokens = word_tokenize(query.lower())  # Use word_tokenize for proper splitting
        result = None
        operator = None
        all_docs = set(range(num_docs))  # Set of all document IDs
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in operators:
                operator = token
                i += 1
                continue
            term = ps.stem(token)
            current_docs = set(inverted_index.get(term, []))
            if operator == "not":
                current_docs = all_docs - current_docs
                operator = None
            if result is None:
                result = current_docs
            elif operator == "or":
                result = result.union(current_docs)
                operator = None
            elif operator == "and":
                result = result.intersection(current_docs)
                operator = None
            i += 1
        return result if result is not None else set()

    # Process query
    result = query_processing(query, inverted_index, len(docs))

    # Evaluate results
    metrics = evaluate_query(result, relevant_docs)

    # Print results
    print("Documents:", docs)
    print("Labels:", labels)
    print("Preprocessed Documents:", preprocess_docs)
    print("Preprocessed Query (terms only):", preprocess_query)
    print("Inverted Index:", dict(inverted_index))
    print("Query:", query)
    print("Relevant Document IDs (from labels):", relevant_docs)
    print("Retrieved Document IDs:", result)
    print("Retrieved Documents:", [docs[i] for i in result])
    print("Evaluation Metrics:", metrics)

# Example usage
if __name__ == "__main__":
    # Write sample data to data.txt for testing
    sample_data = """Information retrieval is fun.,R
Retrieval of information.,R
Fun with information.,NR
Fun is always fun.,NR"""
    with open("data.txt", "w") as f:
        f.write(sample_data)

    file_path = "data.txt"
    query = "information"
    main(file_path, query)

#Bim Kani

import re

file_content = """Document 1
information retreival is good .
search is good
Document 2
learn something useful
Document 3
marry a good guy.life will all like whatt
"""
with open('corpus.txt', 'w') as f:
    f.write(file_content)
print("File created.")

current_doc_lines=[]
docs=[]

def preprocess(text):
    lower_text=text.lower()
    new_string = re.sub(r'[^\w\s]', '', lower_text)
    tokens = new_string.split()
    processed_tokens = []
    for t in tokens:
        if t.isalpha() and t not in stop_words:
            processed_tokens.append(ps.stem(t))
    return processed_tokens

with open('corpus.txt', 'r') as f:
        for line in f:
            if re.match(r'^Document \d+', line):
                if current_doc_lines:
                    docs.append(" ".join(current_doc_lines))

                current_doc_lines = []
            else:
                stripped_line = line.strip()
                if stripped_line:  # Avoid adding empty lines
                    current_doc_lines.append(stripped_line)
if current_doc_lines:
        docs.append(" ".join(current_doc_lines))

print(docs)

query = input("Enter your query: ")
ground_truth_input = input("Enter ground truth relevant documents (e.g., d1,d2): ")

if ground_truth_input:
    true_relevant = set(ground_truth_input.split(','))
else:
    true_relevant = {'d1', 'd2'}

print(true_relevant)
# Preprocess documents and query
processed_docs = [preprocess(d) for d in docs]
processed_query = preprocess(query)

# Unique terms
terms = sorted(set([t for doc in processed_docs for t in doc]))

print("Processed docs:", processed_docs)
print("Processed query:", processed_query)
print("Unique terms:", terms)
print("Ground truth relevant:", true_relevant)

# Binary Independence Model
N = len(docs)

# document frequencies
df_dict = {}

for term in terms:
    df_count = 0
    for doc in processed_docs:
        if term in doc:
            df_count += 1
    df_dict[term] = df_count

bim_idf = {}

for term in terms:
    df = df_dict[term]

    # Calculate the BIM score using the formula
    score = log((N - df + 0.5) / (df + 0.5))

    # Assign the score to the term
    bim_idf[term] = score

rsv_scores = []
for j in range(N):
    score = 0.0
    for q in processed_query:
        if q in processed_docs[j]:
            score += bim_idf[q]
    rsv_scores.append(score)

print("\nBIM RSV scores:", rsv_scores)

RSV_THRESHOLD = 0
relevant_bim = []
for j in range(N):
    if rsv_scores[j] < RSV_THRESHOLD:
        relevant_bim.append(f"d{j+1}")
print(f"BIM relevant documents (RSV < {RSV_THRESHOLD}):", relevant_bim)

retrieved = set(relevant_bim)
if retrieved:
    precision = len(retrieved & true_relevant) / len(retrieved)
else:
    precision = 0

if true_relevant:
    recall = len(retrieved & true_relevant) / len(true_relevant)
else:
    recall = 0

if (precision + recall) > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0
print(f"BIM - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

K = 2
doc_scores = []
for j in range(N):
    doc_tuple = (f"d{j+1}", rsv_scores[j])
    doc_scores.append(doc_tuple)
doc_scores.sort(key=lambda x: x[1], reverse=True)

top_k_slice = doc_scores[:K]
top_k = []
for doc, score in top_k_slice:
    top_k.append(doc)

if K > 0:
    precision_at_k = len(set(top_k) & true_relevant) / K
else:
    precision_at_k = 0
print(f"BIM - Precision@{K}: {precision_at_k:.4f}")

average_precision = 0
relevant_count = 0
for i, (doc, _) in enumerate(doc_scores, 1):
    if doc in true_relevant:
        relevant_count += 1
        average_precision += relevant_count / i
if relevant_count > 0:
    average_precision /= len(true_relevant)
print(f"BIM - MAP: {average_precision:.4f}")

incidence_matrix = np.zeros((len(terms), N), dtype=int)
for i, term in enumerate(terms):
    for j, doc in enumerate(processed_docs):
        if term in doc:
            incidence_matrix[i, j] = 1

doc_labels = []
for i in range(N):
    doc_labels.append(f"d{i+1}")

plt.figure(figsize=(8, 5))
plt.bar(doc_labels, rsv_scores, color='lightcoral')
plt.axhline(y=RSV_THRESHOLD, color='b', linestyle='--', label=f'Threshold ({RSV_THRESHOLD})')
plt.xlabel('Documents')
plt.ylabel('RSV Score')
plt.title('BIM RSV Scores for Documents')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(incidence_matrix, xticklabels=doc_labels, yticklabels=terms, cmap='Blues', annot=True)
plt.title('Incidence Matrix Heatmap')
plt.xlabel('Documents')
plt.ylabel('Terms')
plt.show()
N = len(docs) # Total number of documents (Nd)

# --- Step 1: Calculate Document Frequencies (df or dk) ---
df_dict = {}
for term in terms:
    df_count = 0
    for doc in processed_docs:
        if term in doc:
            df_count += 1
    df_dict[term] = df_count

# --- Step 2: Calculate Relevance-Based Components (Nr, rk) ---
Nr = len(true_relevant)
rk_dict = {}
for term in terms:
    rk_count = 0
    # Check only the relevant documents
    for doc_id_str in true_relevant:
        doc_index = int(doc_id_str[1:]) - 1 # Convert 'd1' to index 0
        if doc_index < len(processed_docs) and term in processed_docs[doc_index]:
            rk_count += 1
    rk_dict[term] = rk_count

# --- Step 3: Calculate Probabilities (pk, qk) with Smoothing ---
p_k_dict = {}
q_k_dict = {}
for term in terms:
    rk = rk_dict[term]
    dk = df_dict[term]
    # Smoothed probability of term k in a relevant doc
    p_k = (rk + 0.5) / (Nr + 1)
    # Smoothed probability of term k in a non-relevant doc
    q_k = (dk - rk + 0.5) / (N - Nr + 1)
    p_k_dict[term] = p_k
    q_k_dict[term] = q_k

# --- Step 4: Calculate Term Relevance Weights (trk) ---
tr_k_weights = {}
for term in terms:
    p_k = p_k_dict[term]
    q_k = q_k_dict[term]
    # Formula: log( (p_k * (1-q_k)) / (q_k * (1-p_k)) )
    # This weight is the trk value for each term
    if p_k > 0 and q_k > 0 and p_k < 1 and q_k < 1:
         weight = log((p_k * (1 - q_k)) / (q_k * (1 - p_k)))
    else:
         weight = 0 # Assign a neutral weight in edge cases
    tr_k_weights[term] = weight

print("\n--- BIM Model Results ---")
print("Term Relevance Weights (trk):")
for term, weight in tr_k_weights.items():
    if term in processed_query: # Only show weights for query terms
        print(f"  - tr('{term}'): {weight:.4f}")


# --- Step 5: Calculate Final RSV Scores ---
rsv_scores = []
for doc in processed_docs:
    score = 0.0
    for q_term in processed_query:
        if q_term in doc:
            score += tr_k_weights[q_term] # Sum the trk weights
    rsv_scores.append(score)

print("\nBIM RSV scores:", [round(s, 4) for s in rsv_scores])

RSV_THRESHOLD = 0.0
relevant_bim = []
for j in range(N):
    if rsv_scores[j] > RSV_THRESHOLD: # Relevant if score is positive
        relevant_bim.append(f"d{j+1}")
print(f"BIM relevant documents (RSV > {RSV_THRESHOLD}):", relevant_bim)

retrieved = set(relevant_bim)
if retrieved:
    precision = len(retrieved & true_relevant) / len(retrieved)
else:
    precision = 0

if true_relevant:
    recall = len(retrieved & true_relevant) / len(true_relevant)
else:
    recall = 0

if (precision + recall) > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0
print(f"BIM - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

K = 2
doc_scores = []
for j in range(N):
    doc_tuple = (f"d{j+1}", rsv_scores[j])
    doc_scores.append(doc_tuple)
doc_scores.sort(key=lambda x: x[1], reverse=True)

top_k_slice = doc_scores[:K]
top_k = []
for doc, score in top_k_slice:
    top_k.append(doc)

if K > 0:
    precision_at_k = len(set(top_k) & true_relevant) / K
else:
    precision_at_k = 0
print(f"BIM - Precision@{K}: {precision_at_k:.4f}")

average_precision = 0
relevant_count = 0
for i, (doc, _) in enumerate(doc_scores, 1):
    if doc in true_relevant:
        relevant_count += 1
        average_precision += relevant_count / i
if relevant_count > 0:
    average_precision /= len(true_relevant)
print(f"BIM - MAP: {average_precision:.4f}")

incidence_matrix = np.zeros((len(terms), N), dtype=int)
for i, term in enumerate(terms):
    for j, doc in enumerate(processed_docs):
        if term in doc:
            incidence_matrix[i, j] = 1

doc_labels = []
for i in range(N):
    doc_labels.append(f"d{i+1}")

plt.figure(figsize=(8, 5))
plt.bar(doc_labels, rsv_scores, color='skyblue')
plt.axhline(y=RSV_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({RSV_THRESHOLD})')
plt.xlabel('Documents')
plt.ylabel('RSV Score')
plt.title('BIM RSV Scores for Documents (with Relevance Feedback)')
plt.legend()
plt.show()

#bim bhars
import pandas as pd
import nltk
import numpy as np
from math import log
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Initialize PorterStemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocess function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(t) for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

# Read documents and labels from a text file
def read_documents(file_path):
    try:
        # Read CSV file with text and label columns
        df = pd.read_csv(file_path, header=0, names=['text', 'label'])
        # Validate labels
        df['label'] = df['label'].str.strip().str.upper()
        if not all(df['label'].isin(['R', 'NR'])):
            raise ValueError("Labels must be 'R' or 'NR'")
        return df['text'].tolist(), df['label'].tolist()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return [], []
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], []

# Evaluation metrics
def evaluate_query(retrieved, relevant):
    retrieved = set(retrieved)
    relevant = set(relevant)
    true_positives = len(retrieved & relevant)
    precision = true_positives / len(retrieved) if retrieved else 0.0
    recall = true_positives / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1_score': f1}

# Main processing function
def main(file_path, query):
    # Read documents and labels
    docs, labels = read_documents(file_path)
    if not docs:
        return

    # Get ground truth: documents labeled 'R'
    true_relevant = {f"d{i+1}" for i, label in enumerate(labels) if label == 'R'}

    # Preprocess documents and query
    processed_docs = [preprocess(d) for d in docs]
    processed_query = preprocess(query)

    # Unique terms
    terms = sorted(set(t for doc in processed_docs for t in doc))

    # Binary Independence Model (BIM) - First Phase
    N = len(docs)
    df_dict = {term: sum(1 for doc in processed_docs if term in doc) for term in terms}
    bim_idf = {term: log((N - df_dict[term] + 0.5) / (df_dict[term] + 0.5)) for term in terms}

    rsv_scores = []
    query_terms = processed_query
    for j in range(N):
        score = sum(bim_idf.get(q, 0) for q in query_terms if q in processed_docs[j])
        rsv_scores.append(score)

    # Relevant: docs with RSV != 0
    RSV_THRESHOLD = 0
    bim_relevant = [f"d{j+1}" for j in range(N) if rsv_scores[j] != 0]

    # Evaluation metrics for first phase
    bim_metrics = evaluate_query(bim_relevant, true_relevant)

    # Precision@K (K=2) for first phase
    K = 2
    doc_scores = [(f"d{j+1}", rsv_scores[j]) for j in range(N)]
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_k = [doc for doc, score in doc_scores[:K]]
    precision_at_k = len(set(top_k) & true_relevant) / K if K > 0 else 0

    # MAP for first phase
    average_precision = 0
    relevant_count = 0
    for i, (doc, _) in enumerate(doc_scores, 1):
        if doc in true_relevant:
            relevant_count += 1
            average_precision += relevant_count / i
    if relevant_count > 0:
        average_precision /= len(true_relevant)

    # Second Phase: Re-ranking with TF-IDF
    # Compute term frequency (TF) for each query term in each document
    tf_dict = defaultdict(lambda: defaultdict(int))
    for j, doc in enumerate(processed_docs):
        for term in doc:
            tf_dict[term][j] += 1

    # Compute TF-IDF scores
    tfidf_scores = []
    for j in range(N):
        score = sum(bim_idf.get(q, 0) * tf_dict[q][j] for q in query_terms if q in processed_docs[j])
        tfidf_scores.append(score)

    # Relevant: docs with TF-IDF != 0
    tfidf_relevant = [f"d{j+1}" for j in range(N) if tfidf_scores[j] != 0]

    # Evaluation metrics for second phase
    tfidf_metrics = evaluate_query(tfidf_relevant, true_relevant)

    # Precision@K (K=2) for second phase
    tfidf_doc_scores = [(f"d{j+1}", tfidf_scores[j]) for j in range(N)]
    tfidf_doc_scores.sort(key=lambda x: x[1], reverse=True)
    tfidf_top_k = [doc for doc, score in tfidf_doc_scores[:K]]
    tfidf_precision_at_k = len(set(tfidf_top_k) & true_relevant) / K if K > 0 else 0

    # MAP for second phase
    tfidf_average_precision = 0
    tfidf_relevant_count = 0
    for i, (doc, _) in enumerate(tfidf_doc_scores, 1):
        if doc in true_relevant:
            tfidf_relevant_count += 1
            tfidf_average_precision += tfidf_relevant_count / i
    if tfidf_relevant_count > 0:
        tfidf_average_precision /= len(true_relevant)

    # Incidence Matrix for Heatmap
    incidence_matrix = np.zeros((len(terms), N), dtype=int)
    for i, term in enumerate(terms):
        for j, doc in enumerate(processed_docs):
            if term in doc:
                incidence_matrix[i, j] = 1

    # Print results
    print("\nDocuments:", docs)
    print("Labels:", labels)
    print("Processed Documents:", processed_docs)
    print("Processed Query:", processed_query)
    print("Unique Terms:", terms)
    print("\nFirst Phase (BIM):")
    print("RSV Scores:", rsv_scores)
    print(f"Relevant Documents (RSV != {RSV_THRESHOLD}):", bim_relevant)
    print("Relevant Document Texts:", [docs[int(doc[1:])-1] for doc in bim_relevant])
    print("Ground Truth Relevant:", true_relevant)
    print("BIM Metrics:", bim_metrics)
    print(f"Precision@{K}: {precision_at_k:.4f}")
    print(f"MAP: {average_precision:.4f}")
    print("\nSecond Phase (TF-IDF Re-ranking):")
    print("TF-IDF Scores:", tfidf_scores)
    print(f"Relevant Documents (TF-IDF != {RSV_THRESHOLD}):", tfidf_relevant)
    print("Relevant Document Texts:", [docs[int(doc[1:])-1] for doc in tfidf_relevant])
    print("TF-IDF Metrics:", tfidf_metrics)
    print(f"Precision@{K}: {tfidf_precision_at_k:.4f}")
    print(f"MAP: {tfidf_average_precision:.4f}")

    # Visualizations
    # Bar Plot for RSV and TF-IDF Scores
    plt.figure(figsize=(10, 6))
    x = np.arange(N)
    width = 0.35
    plt.bar(x - width/2, rsv_scores, width, label='BIM RSV Scores', color='lightcoral')
    plt.bar(x + width/2, tfidf_scores, width, label='TF-IDF Scores', color='skyblue')
    plt.axhline(y=RSV_THRESHOLD, color='b', linestyle='--', label=f'Threshold ({RSV_THRESHOLD})')
    plt.xlabel('Documents')
    plt.ylabel('Scores')
    plt.title('BIM vs TF-IDF Scores for Documents')
    plt.xticks(x, [f"d{i+1}" for i in range(N)])
    plt.legend()
    plt.show()

    # Heatmap for Incidence Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(incidence_matrix, xticklabels=[f"d{i+1}" for i in range(N)], yticklabels=terms, cmap='Blues', annot=True)
    plt.title('Incidence Matrix Heatmap')
    plt.xlabel('Documents')
    plt.ylabel('Terms')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Write sample data to data.txt for testing
    sample_data = """text,label
Information retrieval is fun.,R
Retrieval of information.,R
Fun with information.,NR
Fun is always fun.,NR"""
    with open("data.txt", "w") as f:
        f.write(sample_data)

    file_path = "data.txt"
    query = "information"  # Simulating user input
    main(file_path, query)

#vector space model

import pandas as pd
import nltk
import numpy as np
from math import log
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocess function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(t) for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

# Read documents and labels from a text file
def read_documents(file_path):
    try:
        # Read CSV file with text and label columns
        df = pd.read_csv(file_path, header=0, names=['text', 'label'])
        # Validate labels
        df['label'] = df['label'].str.strip().str.upper()
        if not all(df['label'].isin(['R', 'NR'])):
            raise ValueError("Labels must be 'R' or 'NR'")
        return df['text'].tolist(), df['label'].tolist()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return [], []
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], []

# Evaluation metrics
def evaluate_query(retrieved, relevant):
    retrieved = set(retrieved)
    relevant = set(relevant)
    true_positives = len(retrieved & relevant)
    precision = true_positives / len(retrieved) if retrieved else 0.0
    recall = true_positives / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1_score': f1}

# Calculate Jaccard and Dice similarities
def jaccard_similarity(doc_terms, query_terms):
    doc_set = set(doc_terms)
    query_set = set(query_terms)
    intersection = len(doc_set & query_set)
    union = len(doc_set | query_set)
    return intersection / union if union > 0 else 0.0

def dice_similarity(doc_terms, query_terms):
    doc_set = set(doc_terms)
    query_set = set(query_terms)
    intersection = len(doc_set & query_set)
    return (2 * intersection) / (len(doc_set) + len(query_set)) if (len(doc_set) + len(query_set)) > 0 else 0.0

# Main processing function
def main(file_path):
    # Read documents and labels
    docs, labels = read_documents(file_path)
    if not docs:
        return

    # Get ground truth: documents labeled 'R'
    true_relevant = {f"d{i+1}" for i, label in enumerate(labels) if label == 'R'}

    # Ask user for query
    query = input("Enter your query (e.g., 'information retrieval'): ")

    # Preprocess documents and query
    processed_docs = [preprocess(d) for d in docs]
    processed_query = preprocess(query)

    # Unique terms
    terms = sorted(set(t for doc in processed_docs for t in doc))

    # Vector Space Model (VSM) - Cosine Similarity
    N = len(docs)
    tfidf_docs = np.zeros((len(terms), N))
    for i, term in enumerate(terms):
        df = sum(1 for doc in processed_docs if term in doc)
        idf = log(N / df) if df > 0 else 0
        for j, doc in enumerate(processed_docs):
            tf = doc.count(term)
            tfidf_docs[i, j] = tf * idf

    # Query TF-IDF
    tfidf_query = np.zeros(len(terms))
    for i, term in enumerate(terms):
        tf = processed_query.count(term)
        df = sum(1 for doc in processed_docs if term in doc)
        idf = log(N / df) if df > 0 else 0
        tfidf_query[i] = tf * idf

    # Cosine similarity
    cosine_similarities = []
    for j in range(N):
        doc_vec = tfidf_docs[:, j]
        if np.linalg.norm(doc_vec) > 0 and np.linalg.norm(tfidf_query) > 0:
            cos = np.dot(doc_vec, tfidf_query) / (np.linalg.norm(doc_vec) * np.linalg.norm(tfidf_query))
        else:
            cos = 0
        cosine_similarities.append(cos)

    # Jaccard and Dice similarities
    jaccard_similarities = [jaccard_similarity(doc, processed_query) for doc in processed_docs]
    dice_similarities = [dice_similarity(doc, processed_query) for doc in processed_docs]

    # Relevant documents: similarity > 0
    SIM_THRESHOLD = 0
    cosine_relevant = [f"d{j+1}" for j in range(N) if cosine_similarities[j] > SIM_THRESHOLD]
    jaccard_relevant = [f"d{j+1}" for j in range(N) if jaccard_similarities[j] > SIM_THRESHOLD]
    dice_relevant = [f"d{j+1}" for j in range(N) if dice_similarities[j] > SIM_THRESHOLD]

    # Evaluation metrics
    cosine_metrics = evaluate_query(cosine_relevant, true_relevant)
    jaccard_metrics = evaluate_query(jaccard_relevant, true_relevant)
    dice_metrics = evaluate_query(dice_relevant, true_relevant)

    # Precision@K (K=2) and MAP for each similarity measure
    K = 2
    # Cosine
    cosine_doc_scores = [(f"d{j+1}", cosine_similarities[j]) for j in range(N)]
    cosine_doc_scores.sort(key=lambda x: x[1], reverse=True)
    cosine_top_k = [doc for doc, score in cosine_doc_scores[:K]]
    cosine_precision_at_k = len(set(cosine_top_k) & true_relevant) / K if K > 0 else 0
    cosine_ap = 0
    relevant_count = 0
    for i, (doc, _) in enumerate(cosine_doc_scores, 1):
        if doc in true_relevant:
            relevant_count += 1
            cosine_ap += relevant_count / i
    cosine_map = cosine_ap / len(true_relevant) if relevant_count > 0 else 0

    # Jaccard
    jaccard_doc_scores = [(f"d{j+1}", jaccard_similarities[j]) for j in range(N)]
    jaccard_doc_scores.sort(key=lambda x: x[1], reverse=True)
    jaccard_top_k = [doc for doc, score in jaccard_doc_scores[:K]]
    jaccard_precision_at_k = len(set(jaccard_top_k) & true_relevant) / K if K > 0 else 0
    jaccard_ap = 0
    relevant_count = 0
    for i, (doc, _) in enumerate(jaccard_doc_scores, 1):
        if doc in true_relevant:
            relevant_count += 1
            jaccard_ap += relevant_count / i
    jaccard_map = jaccard_ap / len(true_relevant) if relevant_count > 0 else 0

    # Dice
    dice_doc_scores = [(f"d{j+1}", dice_similarities[j]) for j in range(N)]
    dice_doc_scores.sort(key=lambda x: x[1], reverse=True)
    dice_top_k = [doc for doc, score in dice_doc_scores[:K]]
    dice_precision_at_k = len(set(dice_top_k) & true_relevant) / K if K > 0 else 0
    dice_ap = 0
    relevant_count = 0
    for i, (doc, _) in enumerate(dice_doc_scores, 1):
        if doc in true_relevant:
            relevant_count += 1
            dice_ap += relevant_count / i
    dice_map = dice_ap / len(true_relevant) if relevant_count > 0 else 0

    # Print results
    print("\nDocuments:", docs)
    print("Labels:", labels)
    print("Processed Documents:", processed_docs)
    print("Processed Query:", processed_query)
    print("Unique Terms:", terms)
    print("\nVSM Retrieval (Cosine Similarity):")
    print("Cosine Similarities:", cosine_similarities)
    print(f"Relevant Documents (Similarity > {SIM_THRESHOLD}):", cosine_relevant)
    print("Relevant Document Texts:", [docs[int(doc[1:])-1] for doc in cosine_relevant])
    print("Cosine Metrics:", cosine_metrics)
    print(f"Cosine Precision@{K}: {cosine_precision_at_k:.4f}")
    print(f"Cosine MAP: {cosine_map:.4f}")
    print("\nVSM Retrieval (Jaccard Similarity):")
    print("Jaccard Similarities:", jaccard_similarities)
    print(f"Relevant Documents (Similarity > {SIM_THRESHOLD}):", jaccard_relevant)
    print("Relevant Document Texts:", [docs[int(doc[1:])-1] for doc in jaccard_relevant])
    print("Jaccard Metrics:", jaccard_metrics)
    print(f"Jaccard Precision@{K}: {jaccard_precision_at_k:.4f}")
    print(f"Jaccard MAP: {jaccard_map:.4f}")
    print("\nVSM Retrieval (Dice Similarity):")
    print("Dice Similarities:", dice_similarities)
    print(f"Relevant Documents (Similarity > {SIM_THRESHOLD}):", dice_relevant)
    print("Relevant Document Texts:", [docs[int(doc[1:])-1] for doc in dice_relevant])
    print("Dice Metrics:", dice_metrics)
    print(f"Dice Precision@{K}: {dice_precision_at_k:.4f}")
    print(f"Dice MAP: {dice_map:.4f}")
    print("\nGround Truth Relevant:", true_relevant)

    # Visualizations
    # Bar Plot for Cosine Similarities
    plt.figure(figsize=(8, 5))
    plt.bar([f"d{i+1}" for i in range(N)], cosine_similarities, color='skyblue')
    plt.axhline(y=SIM_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({SIM_THRESHOLD})')
    plt.xlabel('Documents')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity of Documents to Query')
    plt.legend()
    plt.show()

    # Bar Plot for Jaccard Similarities
    plt.figure(figsize=(8, 5))
    plt.bar([f"d{i+1}" for i in range(N)], jaccard_similarities, color='lightgreen')
    plt.axhline(y=SIM_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({SIM_THRESHOLD})')
    plt.xlabel('Documents')
    plt.ylabel('Jaccard Similarity')
    plt.title('Jaccard Similarity of Documents to Query')
    plt.legend()
    plt.show()

    # Bar Plot for Dice Similarities
    plt.figure(figsize=(8, 5))
    plt.bar([f"d{i+1}" for i in range(N)], dice_similarities, color='lightcoral')
    plt.axhline(y=SIM_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({SIM_THRESHOLD})')
    plt.xlabel('Documents')
    plt.ylabel('Dice Similarity')
    plt.title('Dice Similarity of Documents to Query')
    plt.legend()
    plt.show()

    # Heatmap for TF-IDF Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(tfidf_docs, xticklabels=[f"d{i+1}" for i in range(N)], yticklabels=terms, cmap='YlGnBu', annot=True)
    plt.title('TF-IDF Matrix Heatmap')
    plt.xlabel('Documents')
    plt.ylabel('Terms')
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = "data.txt"
    main(file_path)

import pandas as pd
import nltk
import numpy as np
from math import log
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocess function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(t) for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

# Read documents from a text file
def read_documents(file_path):
    try:
        # Read CSV file with text column (assuming format: text,label)
        df = pd.read_csv(file_path, header=0, names=['text', 'label'])
        return df['text'].tolist()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# Main processing function
def main(file_path):
    # Read documents
    docs = read_documents(file_path)
    if not docs:
        return

    # Get query from user
    query = 'information retrieval'

    # Preprocess documents and query
    processed_docs = [preprocess(d) for d in docs]
    processed_query = preprocess(query)

    # Unique terms
    terms = sorted(set(t for doc in processed_docs for t in doc))

    # Vector Space Model (VSM) - TF-IDF
    N = len(docs)
    tfidf_docs = np.zeros((len(terms), N))
    for i, term in enumerate(terms):
        df = sum(1 for doc in processed_docs if term in doc)
        idf = log(N / df) if df > 0 else 0
        for j, doc in enumerate(processed_docs):
            tf = doc.count(term)
            tfidf_docs[i, j] = tf * idf

    # Query TF-IDF
    tfidf_query = np.zeros(len(terms))
    for i, term in enumerate(terms):
        tf = processed_query.count(term)
        df = sum(1 for doc in processed_docs if term in doc)
        idf = log(N / df) if df > 0 else 0
        tfidf_query[i] = tf * idf

    # Cosine similarity
    cosine_similarities = []
    for j in range(N):
        doc_vec = tfidf_docs[:, j]
        if np.linalg.norm(doc_vec) > 0 and np.linalg.norm(tfidf_query) > 0:
            cos = np.dot(doc_vec, tfidf_query) / (np.linalg.norm(doc_vec) * np.linalg.norm(tfidf_query))
        else:
            cos = 0
        cosine_similarities.append(cos)

    # Rank documents by cosine similarity
    doc_scores = [(f"d{j+1}", cosine_similarities[j], docs[j]) for j in range(N)]
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    # Print results
    print("\nDocuments:", docs)
    print("Processed Query:", processed_query)
    print("\nRanked Documents (Cosine Similarity):")
    for doc_id, score, text in doc_scores:
        print(f"{doc_id}: Score = {score:.4f}, Text = {text}")

# Example usage
if __name__ == "__main__":
    file_path = "data.txt"
    main(file_path)

data = pd.read_csv("data.txt",header=0,names=["text","label"])
df = data["text"].tolist()
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(t) for t in tokens if t.isalpha()]
    return tokens

print(preprocess("I love cats"))
