from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def tfidf_match(input_desc_list, target_desc_list):
    assert isinstance(input_desc_list, list)
    assert isinstance(target_desc_list, list)

    # combine input and description lists for consistent TF-IDF vocabulary between queries and corpus
    combined_corpus = input_desc_list + target_desc_list

    # text cleaning was already done, or at least should have been done before using this function
    vectorizer = TfidfVectorizer() # stop_words="english", lowercase=True
    vectorizer.fit(combined_corpus)

    tfidf_input = vectorizer.transform(input_desc_list)   # queries
    tfidf_target = vectorizer.transform(target_desc_list) # corpus

    # compute pairwise cosine similarity between each input vector and all target vectors.
    similarity_matrix = cosine_similarity(tfidf_input, tfidf_target)

    results = []
    for i, row in enumerate(similarity_matrix):
        best_index = np.argmax(row)
        best_score = row[best_index]
        best_match = target_desc_list[best_index]

        # save the best match and its similarity score
        results.append((best_match, best_score))

    results = pd.DataFrame(results, columns=["match_tfidf", "score_tfidf"])
    # results["index"] = [i for i in range(len(results))]

    return results