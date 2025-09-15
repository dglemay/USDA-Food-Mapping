##################################################################################################
# checking_asa24tofoodb.py
# Purpose: this code checkes the results of ASA24 to FooDB against ground truth
# This version expects csv files as inputs.
# 
# Input description supplied 
# Input targets supplied 
# Input ground truth mapping
# Input Claude results
# 
# Output prefix to create 
#   PREFIX_stdout.txt
#   PREFIX_merged_preditions.csv
#   PREFIX_pred_is_wrong.csv
#
# Example usage: 
# python checking_asa24tofoodb.py input_desc_list.csv target_desc_list.csv ASA24_FooDB_groundtruth_input_target.csv matched_haiku_asa24_exp4_results081525.txt haiku_ASA24exp4
# 
# Authors: Michael Strohmeier & Danielle Lemay
#################################################################################################
import sys
import pandas as pd
import numpy as np
import re
import csv

# Check if correct number of arguments provided
if len(sys.argv) != 6:
    print("Usage: python checkingcsv_asa24tofoodb.py <input_file1.csv> <input_file2.csv> <input_file3.csv> <input_file4.csv> <Outputfile_prefix>")
    sys.exit(1)

# Get filenames from command line arguments
input_file = sys.argv[1]
target_file = sys.argv[2]
truth_file = sys.argv[3]
match_file = sys.argv[4]
prefix = sys.argv[5]

# Redirect stdout to file based on prefix
stdout_file = f"{prefix}stdout.txt"
sys.stdout = open(stdout_file, 'w')

# function to read ground truth
def load_input_files(input_file, target_file, truth_file):
    """Load three CSV files and return as DataFrames"""
    try:
        print("Loading input file")
        inputdf = pd.read_csv(input_file, sep="\t", header=0, dtype=str) #first row is header
        print(f"Number of unique inputs {len(inputdf)}")

        print("Loading target file")
        targetdf = pd.read_csv(target_file, sep="\t", header=0, dtype=str) #first row is header
        print(f"Number of unique targets {len(targetdf)}")  

        print("Loading truth file")
        truthdf = pd.read_csv(truth_file, sep="\t", header=0, dtype=str) #first row is header
        print(f"Number of items in ground truth table {len(truthdf)}")

        # turning the food descriptions in each column to lowercase
        inputdf['input_desc'] = inputdf['input_desc'].str.lower()
        targetdf['target_desc'] = targetdf['target_desc'].str.lower()
        truthdf['input_desc'] = truthdf['input_desc'].str.lower()
        truthdf['target_desc'] = truthdf['target_desc'].str.lower()

        return inputdf, targetdf, truthdf
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None, None, None
    except pd.errors.EmptyDataError as e:
        print(f"Empty file error: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None, None

if __name__ == "__main__":

    print("Loading dataframes\n")

    inputdf, targetdf, truthdf = load_input_files(input_file, target_file, truth_file)

    print("Dataframes loaded\n")

    # checking original dataset
    print("Length of ground truth table", len(truthdf))

    # read claude's results
    claude_df = pd.read_csv(match_file, sep="\t", header=None, names=["input_desc", "match_claude"], dtype=str)
    
    # turning the food descriptions in each column to lowercase
    claude_df['input_desc'] = claude_df['input_desc'].str.lower()
    claude_df['match_claude'] = claude_df['match_claude'].str.lower()

    print()
    print("Length of claude table", len(claude_df))
    print() 
    print("Check that every input in Claude was also in ground truth")
    # check that every input in Claude was also in ground truth
    claude_inputs_not_in_ground_truth = []
    for item in inputdf['input_desc']:
        if item not in claude_df['input_desc'].values:
            claude_inputs_not_in_ground_truth.append(item)
            print(f"Item {item} not in ground truth")
    
    print()
    print("If this next part is 0, we did not hallucinate any inputs")
    print("Number of claude inputs not in ground truth:", len(claude_inputs_not_in_ground_truth))
    print()
    print("Check that every output from Claude actually existed in the target database")
    # check that every output from Claude actually existed in the target database 
    claude_outputs_not_in_target_database = []
    for item in claude_df['match_claude']:
        if item not in targetdf['target_desc'].values:
            claude_outputs_not_in_target_database.append(item)
            print(f"Item {item} not in target database")
    
    print()
    print("If this next part is 0, we did not hallucinate any responses not in target database")
    print("Number of claude outputs not in target database:", len(claude_outputs_not_in_target_database))
    print()

    # merge the ground truth and the predictions into one data frame, linking on the input description
    merged_df = pd.merge(claude_df, truthdf, on="input_desc", how="left")
    
    is_match = []
    for i in range(0, len(merged_df)):
        if (merged_df.iloc[i]["match_claude"] == merged_df.iloc[i]["target_desc"]):
            is_match.append(1)
        else:
            is_match.append(0)

    merged_df["is_match"] = is_match
    
    print()
    if (merged_df["is_match"][:100].sum() > 100): print("Accuracy from first 100 matches:", merged_df["is_match"][:100].sum() / 100)
    print("Accuracy (whole dataset)", merged_df["is_match"].sum() / len(merged_df))

    print()

    # output mismatches
    debug_file = f"{prefix}_pred_is_wrong.csv"
    merged_df[merged_df["is_match"] == 0].to_csv(debug_file, index=False)
    # output all predictions
    all_pred_file = f"{prefix}_predictions_merged.csv"
    merged_df.to_csv(all_pred_file, index=False)
