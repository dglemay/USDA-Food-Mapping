##################################################################################################
# semantic_topK_accuracy.py
# Purpose: Given input food descriptions and the target database, the overall goal is to match
# the food descriptions to the food in the target database using text alone.
# This code tests whether the best match from ground truth is in the top K most similar
# semantic matches. K values of 1, 5, 10, 25, 50 are tested.
# 
# Input description: food descriptions to match to a target database
# Input targets: food descriptions in the target database
# Input grouth truth: matches of food descriptions beteween input database and target database as selected by nutrition scientist
#   The grouth truth file is expected to be two-column tab-delimited text file with a header that has columns named
#  as "input_desc" and "target_desc"
# Output filename 
#
# Example usage: 
# python semantic_topK_accuracy.py input_desc_list.csv target_desc_list.csv ground_truth.csv
# 
# Authors: Danielle Lemay & Michael Strohmeier
#################################################################################################
import sys
import os
import pandas as pd
import json
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Check if correct number of arguments provided
if len(sys.argv) != 4:
    print("Usage: python semantic_topK_accuracy.py <input_file1.txt> <input_file2.txt> <input_file3.txt>")
    sys.exit(1)

# Get filenames from command line arguments
input_file = sys.argv[1] # this file should be a single column, each row is a text description followed by \n
target_file = sys.argv[2] # this file should be a single column, each row is a text description followed by \n
truth_file = sys.argv[3] # for the nhanes experiment, truth file = nhanes_dfg2_labels.csv


def load_inputs():
    inputdf = pd.read_csv(input_file, header=0) #header specifies which row contains the header
    # turning the food descriptions in each column to lowercase
    inputdf['input_desc'] = inputdf['input_desc'].str.lower()

    targetdf = pd.read_csv(target_file, header=0) 
    # turning the food descriptions in each column to lowercase
    targetdf['target_desc'] = targetdf['target_desc'].str.lower()

    return inputdf, targetdf

def load_truth():
    df = pd.read_csv(truth_file, header=0)

    df = df[["ingred_desc", "simple_name", "label"]]
    df.columns = ["input_desc", "target_desc", "label"]

    # turning the food descriptions in each column to lowercase
    columns_to_lower = ["input_desc", "target_desc"]
    df[columns_to_lower] = df[columns_to_lower].apply(lambda x: x.str.lower())

    df = df.drop_duplicates(subset="input_desc", keep="first")

    return df


# convert food descriptions into mathematical representations 
# using a General Text Embeddings (GTE) model,
# then compute the simularity of all inputs and targets
# to identify the top n matches in the target database for each input
def embed_match(input_list, target_list):
    assert isinstance(input_list, list)
    assert isinstance(target_list, list)

    model = SentenceTransformer("thenlper/gte-large")
    
    input_vecs = model.encode(input_list, normalize_embeddings=True)
    target_vecs = model.encode(target_list, normalize_embeddings=True)

    sim_matrix = cosine_similarity(input_vecs, target_vecs)

    return sim_matrix

def top_matches(input_list, target_list, sim_matrix, n):
    #print(f"seeking top {n} matches\n")
    results = [
        {
            "input_desc": input_list[i],
            # Get indices of top-n highest scores (sorted descending)
            "top_matches": [{"target": target_list[j], "score": row[j]} for j in row.argsort()[-n:][::-1]]
        }
        for i, row in enumerate(sim_matrix)
    ]

    return results

# given a dataframe of ground truth that contains both input food description
# and its correct (chosen by a nutrition scientist) target description
# as well as the top K matches selected by text simularity,
# compute the accuracy (best match within the top K matches)
def compute_top_k_accuracy(df, top_k_matches, k):
    correct = 0
    total = 0
    wrong_matches = []

    input_to_true_target = dict(zip(df["input_desc"], df["target_desc"]))

    for match in top_k_matches:
        total += 1
        input_desc = match["input_desc"]
        true_target = input_to_true_target.get(input_desc)
        predicted_targets = [m["target"] for m in match["top_matches"][:k]]
        
        if true_target in predicted_targets:
            correct += 1
       # else:
            #if (k==10):
             #   wrong_matches.append((input_desc, true_target))
             #   print(f"Wrong match for input {input_desc} targets {predicted_targets}, true target is {true_target}\n")

    accuracy = correct / total if total > 0 else 0.0

    return accuracy
    
# Test function
def is_dataframe(obj):
    return isinstance(obj, pd.DataFrame)


# Do all the things...
if __name__ == "__main__":
    # Load from files 
    input_df, target_df = load_inputs()
    truth_df = load_truth()

    list_a = input_df['input_desc'].tolist() # convert dataframe to list
    list_b = target_df['target_desc'].tolist() # convert dataframe to list

    #print(f"Generating embeddings, cosine similarity matrix\n")
    smatrix = embed_match(list_a, list_b)
    # report the accuracy for the top K matches
    K_list = (1, 5, 10, 25, 50, 100)
    for K in K_list:
        topK_matches = top_matches(list_a, list_b, smatrix, K)
        topK_match_accuracy = compute_top_k_accuracy(truth_df, topK_matches, K)
        print(f"Top {K} accuracy (whether or not in target db) is {topK_match_accuracy}")

    print()

    #get the subset of the input for which a true database match exists
    truth_subset_df = truth_df[truth_df["label"] == 1]
    list_w_matches = truth_subset_df['input_desc'].tolist() 
    subset_matrix = embed_match(list_w_matches, list_b)
     # report the accuracy for the top K matches
    K_list = (1, 5, 10, 25, 50, 100)
    for K in K_list:
        topK_matches = top_matches(list_w_matches, list_b, subset_matrix, K)
        topK_match_accuracy = compute_top_k_accuracy(truth_subset_df, topK_matches, K)
        print(f"Top {K} accuracy (for subset with true matches) is {topK_match_accuracy}")
    