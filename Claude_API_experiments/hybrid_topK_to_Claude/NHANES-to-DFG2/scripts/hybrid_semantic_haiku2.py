##################################################################################################
# hybrid_semantic_haiku2.py
# Purpose: Given input food descriptions and the target database, the overall goal is to match
# the food descriptions to the food in the target database using text alone.
# This code implements a hybrid strategy that first uses semantic mapping to determine the top K
# matches, then pass only those topK matches along with the input food description to an LLM
# (Claude haiku) to determine the best match.
# This version optimizes inference parameters: temperature to zero, 
# Repetition penalties are not available parameters, but supposedly stateless between API calls.
# 
# Input description: food descriptions to match to a target database
# Input targets: food descriptions in the target database
# Input grouth truth: matches of food descriptions beteween input database and target database as selected by nutrition scientist
# K value = number of top matches to retain from semantic similarity to provide to LLM for final selection
# Output filename = best match for each input description
#
# Example usage: 
# python hybrid_semantic_haiku.py input_desc_list.csv target_desc_list.csv 5 matched_hybrid_haiku_NHANESexp4.txt
# 
# Authors: Danielle Lemay & Michael Strohmeier
#################################################################################################
import sys
import os
import pandas as pd
import numpy as np
import anthropic
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Check if correct number of arguments provided
if len(sys.argv) != 5:
    print("Usage: python hybrid_semantic_haiku.py <input_file1.txt> <input_file2.txt> <input_file3.txt> <Kvalue> <output_file.txt>")
    sys.exit(1)

# Get filenames from command line arguments
input_file = sys.argv[1] # this file should be a single column, each row is a text description followed by \n
target_file = sys.argv[2] # this file should be a single column, each row is a text description followed by \n
K = int(sys.argv[3])
output_file = sys.argv[4]

# Configuration
API_KEY = os.getenv('ANTHROPIC_API_KEY')
MODEL = "claude-3-haiku-20240307"  # Claude Haiku
RATE_LIMIT_DELAY = 1  # seconds between API calls to avoid rate limiting
DEBUG_MODE = 0 # use '1' to debug, '0' for normal run
DEBUG_CLAUDE = 0 # use '1' to debug, '0' for normal run
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=API_KEY)

def load_inputs():
    inputdf = pd.read_csv(input_file, header=0) #header specifies which row contains the header
    # turning the food descriptions in each column to lowercase
    inputdf['input_desc'] = inputdf['input_desc'].str.lower()

    targetdf = pd.read_csv(target_file, header=0) 
    # turning the food descriptions in each column to lowercase
    targetdf['target_desc'] = targetdf['target_desc'].str.lower()

    return inputdf, targetdf

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


# get the top K targets (as a list) for a given index into the input_list
# sim_matrix: Each row represents an input, Each column represents a target
def get_top_k_targets(input_idx, target_list, sim_matrix, k):
    scores = sim_matrix[input_idx] # sim_matrix[i][j] is the similarity between input i and target j
    top_indices = np.argsort(scores)[-k:][::-1]
    return [target_list[i] for i in top_indices]


def find_best_match(input_list, target_list, sim_matrix, n, file_handle) -> str:
    """
    Find the best match for input_list in target_list using Claude Haiku.
    Returns the best match or "none" if no good match is found.
    """
    try:
        # get the top n matches to send to Claude
        for idx in range(len(input_list)):
            top_match_list = get_top_k_targets(idx, target_list, sim_matrix, n)
            if DEBUG_MODE: 
                print(f"Food item is {input_list[idx]}")
                print(f"top {n} matches are {top_match_list}")
            else:
                # Create prompt for Claude
                #list_b_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(list_b)])
        
                prompt = f"""Given the food item: "{input_list[idx]}"

                Find the most nutritionally similar match from this list:
                {top_match_list}

Matching criteria (in order of priority):
1. Same animal or plant source
2. Nutritional profile similarity (macronutrients, micronutrients, calories per serving)
3. Same preparation method
4. Semantic/name similarity

Rules:
- Return ONLY the exact text of the best matching item from the list
- Be STRICT: if there is not a match with the same animal or plant source, return "none"
- Be STRICT: if there is not a match with strong nutritional similarity (80%+ similar nutritionally), return "none"
- Your response must contain ONLY the matching text or "none" - no explanations

Best food match:"""

                response = client.messages.create(
                    model=MODEL,
                    max_tokens=50,
                    temperature=0,
                    system="You are a strict food matching assistant. Return the best matching text from the provided list or 'none'. Do not include additional text.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]   
                )
        
                result = response.content[0].text.strip().strip('"\'')
                if DEBUG_CLAUDE: print(f"food item is {input_list[idx]}")
                if DEBUG_CLAUDE: print(f"result from Claude is {result}")

                match_line = f"{input_list[idx]}\t{result}"
                file_handle.write(match_line + '\n') 
            
    except Exception as e:
        print(f"Error processing '{input_list[idx]}': {e}")
        return "none"
    

# Do all the things...
if __name__ == "__main__":
    # Load from files 
    input_df, target_df = load_inputs()

    list_a = input_df['input_desc'].tolist() # convert dataframe to list
    list_b = target_df['target_desc'].tolist() # convert dataframe to list

    print(f"Generating embeddings, cosine similarity matrix\n")
    smatrix = embed_match(list_a, list_b)
    with open(output_file, 'w') as outf:
        print(f"Find best matches with Claude")
        find_best_match(list_a, list_b, smatrix, K, outf)
    
    print(f"\nDone! Check {output_file} for results.")