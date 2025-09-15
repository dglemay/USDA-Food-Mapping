from rapidfuzz import fuzz, process
import pandas as pd

def fuzzy_match(input_desc_list, target_desc_list):
    assert type(input_desc_list) == list
    assert type(target_desc_list) == list

    results = []
    for input_desc in input_desc_list:
        # fuzz.ratio computes a normalized Levenshtein similarity (edit distance)
        best_match, score, best_match_index = process.extractOne(input_desc, target_desc_list, scorer=fuzz.ratio)
        
        # save the best match and its similarity score
        results.append((best_match, score))

    results = pd.DataFrame(results, columns=["match_fuzzy", "score_fuzzy"])
        
    return results