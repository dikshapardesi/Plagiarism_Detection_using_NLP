import os
import urllib.parse
import json
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

submitted_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
dataset_files = [open(_file, encoding='utf-8').read()
                 for _file in submitted_files]

def vectorize(Text): return TfidfVectorizer().fit_transform(Text).toarray()
def similarity(doc1, doc2): return cosine_similarity([doc1, doc2])

vectors = vectorize(dataset_files)
s_vectors = list(zip(submitted_files, vectors))

def check_plagiarism():
    plagiarism_results = {}
    global s_vectors
    for submission_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((submission_a, text_vector_a))
        del new_vectors[current_index]
        for submission_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            if(sim_score > 0):
                sim_score = round(sim_score, 1)
                submission_pair = sorted((os.path.splitext(submission_a)[0], os.path.splitext(submission_b)[0]))
                res = (submission_pair[0]+' similar to '+ submission_pair[1])
                plagiarism_results[res] = sim_score
    api = json.dumps(plagiarism_results)
    return api

if __name__ == "__main__":
    print(check_plagiarism())
