##################################################################################################
# map_ASA24_to_FooDB.py
# Purpose: Given input food descriptions/IDs from ASA24 data and the target database of 
# food descriptons/IDs (FooDB), the overall goal is to match the food descriptions to the food 
# in the target database.
# 
# This code stitches together results files for the following strategy:
#  If food IDs match 1-to-1, make the match 
#  If food IDs are 1-to-many, collect the match from an LLM
#  If there is no food ID match, use semantic matching to identify top K matches, send those to LLM
# 
# Example usage: 
# python map_ASA24_to_FooDB.py 
# 
# Authors: Danielle Lemay 
######################################################################################################

import pandas as pd

input1to1_match_file = "../data/processed/input_1to1_ASA24toFooDB.txt"
one_to_many_match_file = "../results/matched_1toMany_Sonnet.txt"
no_id_match_file = "../results/matched_FooDB_hybrid_Sonnet_k5.txt"
target_file = "../data/processed/target_desc_FooDB.txt"
output_file = "../results/complete_map_ASA24toFooDB.txt"
try:
    # Load all of the files
    print(f"Stitching all the dataframes together\n")
    input1to1_df = pd.read_csv(input1to1_match_file, sep="\t", header=0) # columns are input_id, input_desc
    target_df = pd.read_csv(target_file, sep="\t", header=0) # columns are target_id, target_desc

    # where the IDs match 1-to-1, make the match
    input_to_target_df = pd.merge(input1to1_df, target_df, left_on='input_id', right_on='target_id')
    match1to1_df = input_to_target_df[['input_desc','target_desc']]
    match1to1_df = match1to1_df.rename(columns={'target_desc': 'predicted_match'})
    match1to1_subset_df = match1to1_df[['input_desc','predicted_match']]

    # get 1-to-Many ID matches from Claude Sonnet
    match1toM_df = pd.read_csv(one_to_many_match_file, sep="\t", header=None, names=['input_desc', 'predicted_match'])
    # get matches from Claude Sonnet when there was no ID match
    matchNoID_df = pd.read_csv(no_id_match_file, sep="\t", header=None, names=['input_desc', 'predicted_match'])

    all_matches = pd.concat([match1to1_subset_df, match1toM_df, matchNoID_df], ignore_index=True)

    # write to an output file
    all_matches.to_csv(output_file, sep='\t', index=False, header=True)

    print(f"\nDone! Check {output_file} for results.")

except Exception as e:
    # handle any exception and access the exception details
    print(f"An error occurred: {e}")