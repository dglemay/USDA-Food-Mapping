from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def embed_match(input_list, target_list):
    assert isinstance(input_list, list)
    assert isinstance(target_list, list)

    # load a pre-trained sentence embedding model from HuggingFace.
    # "thenlper/gte-large" read about it here:
    # https://huggingface.co/thenlper/gte-large
    model = SentenceTransformer("thenlper/gte-large")

    # embedding text describtions
    # normalize embeddings so each vector has unit length 1
    input_vecs = model.encode(input_list, normalize_embeddings=True)
    target_vecs = model.encode(target_list, normalize_embeddings=True)

    # compute pairwise cosine similarity between each input vector and all target vectors.
    sim_matrix = cosine_similarity(input_vecs, target_vecs)

    results = []
    for i, row in enumerate(sim_matrix):
        # for each input, find the target with the highest cosine similarity score
        best_index = row.argmax()
        best_score = row[best_index]
        best_match = target_list[best_index]

        # save the best match and its similarity score
        results.append((best_match, best_score))

    df_results = pd.DataFrame(results, columns=["match_embed", "score_embed"])
    
    return df_results
