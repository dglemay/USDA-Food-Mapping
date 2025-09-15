##################################################################################################
# choose_from_1toMany_Claude.py
# Purpose: Given input food ID/descriptions and the target database of food ID/descriptions, 
# where there are 1-to-Many input ID to target ID matches database matches, 
# choose the best match for the input food.
# 
# For each input ID, send all target ID matches to LLM to make the choice
# 
# input.txt = ID and text descriptions from input database 
# target.txt = ID and text descriptions from target database 
# Model = which LLM to use (currently supports Claude models. answer must be "Haiku" or "Sonnet")
# Output filename = best match for each input description
#
# Example usage: 
# python choose_from_1toMany_Claude.py input.txt target.txt Haiku matched_1toMany_Haiku.txt
# 
# Authors: Danielle Lemay & Michael Strohmeier
#################################################################################################
import sys
import os
import pandas as pd
import numpy as np
import anthropic
#from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Check if correct number of arguments provided
if len(sys.argv) != 5:
    print("Usage: python hybrid_semantic_Claude.py input.txt target.txt Haiku matched_FooDB_hybrid_Haiku_k10.txt")
    sys.exit(1)

# Get filenames from command line arguments
input_file = sys.argv[1]
target_file = sys.argv[2]
modelname = sys.argv[3]
output_file = sys.argv[4]

# Configuration
API_KEY = os.getenv('ANTHROPIC_API_KEY')
RATE_LIMIT_DELAY = 1  # seconds between API calls to avoid rate limiting
DEBUG_MODE = 0 # use '1' to debug, '0' for normal run
DEBUG_CLAUDE = 0 # use '1' to debug, '0' for normal run
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if (modelname == "Haiku"):
    MODEL = "claude-3-haiku-20240307"  # Claude Haiku
elif (modelname == "Sonnet"):
    MODEL = "claude-sonnet-4-20250514"  # Claude Sonnet 4"
else :
    raise Exception("Model must be Haiku or Sonnet")

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=API_KEY)


# find best match for input_item in target_list using Claude and write to output file
def find_best_match_for_item(input_item, target_list, file_handle):
    try:
        # Create prompt for Claude
        prompt = f"""Given the food item: "{input_item}"

        Find the most nutritionally similar match from this list:
        {target_list}

Matching criteria (in order of priority):
1. Same animal or plant source
2. Nutritional profile similarity (macronutrients, micronutrients, calories per serving)
3. Same preparation method
4. Semantic/name similarity

Rules:
- Return ONLY the exact text of the best matching item from the list
- Your response must contain ONLY the matching text - no explanations

Best food match:"""

        response = client.messages.create(
            model=MODEL,
            max_tokens=50,
            temperature=0,
            system="You are a strict food matching assistant. Return the best matching text from the provided list. Do not include additional text.",
            messages=[
                {"role": "user", "content": prompt}
               ]   
            )
        
        result = response.content[0].text.strip().strip('"\'')
        if DEBUG_CLAUDE: print(f"food item is {input_item}")
        if DEBUG_CLAUDE: print(f"result from Claude is {result}")

        match_line = f"{input_item}\t{result}"
        file_handle.write(match_line + '\n') 
            
    except Exception as e:
        print(f"Error processing '{input_item}': {e}")
    

# Do all the things...
try:
    print(f"Loading input files\n")
    input_df = pd.read_csv(input_file, sep="\t", header=0)
    target_df = pd.read_csv(target_file, sep="\t", header=0)

    # get list of food IDs
    input_id_list = input_df['input_id'].tolist() 

    # open output file for writing
    file_handle = open(output_file, "w")
    
    print(f"For each food ID, find best matches with Claude")
    for id in input_id_list:
        item_a = input_df[input_df['input_id'] == id]['input_desc'].iloc[0]
        list_b = target_df[target_df['target_id'] == id]['target_desc'].tolist()
        find_best_match_for_item(item_a, list_b, file_handle)

    file_handle.close()

    print(f"\nDone! Check {output_file} for results.")

except Exception as e:
    # handle any exception and access the exception details
    print(f"An error occurred: {e}")
