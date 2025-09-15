##################################################################################################
# checking_asa24_exp4.py
# Purpose: this code checkes the results of ASA24 experiment 4 against ground truth
# 
# Input description supplied 
# Input targets supplied 
# Input ground truth mapping
# Input, Claude results: 
# Output prefix to create 
#   PREFIX_stdout.txt
#   PREFIX_merged_preditions.csv
#   PREFIX_pred_is_wrong.csv
#
# Example usage: 
# python checking_asa24_exp4.py input_desc_list_noquotes.txt target_desc_list_noquotes.txt ASA24_FooDB_groundtruth_input_target.txt matched_haiku_asa24_exp4_results081525.txt haiku_ASA24exp4
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
    print("Usage: python check_nhanes_exp_4.py <input_file1.csv> <input_file2.csv> <input_file3.csv> <input_file4.csv> <Outputfile_prefix>")
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
        print("Loading input file\n")
        with open(input_file, 'r') as f:
            lines = [line.strip() for line in f]
        inputdf = pd.DataFrame(lines)
        
        print("Loading target file\n")
        with open(target_file, 'r') as f2:
            lines2 = [line2.strip() for line2 in f2]
        targetdf = pd.DataFrame(lines2)

        print("Loading truth file\n")
        truthdf = pd.read_csv(truth_file, sep="\t", header=0) #first row is header

        print("Number of unique inputs", len(inputdf))
        print("Number of unique targets", len(targetdf))

        #truthdf = truthdf["input_desc", "target_desc"]
        #truthdf.columns = ["input_desc", "target_desc"]

        # turning the food descriptions in each column to lowercase
        columns_to_lower = ["input_desc", "target_desc"]
        truthdf[columns_to_lower] = truthdf[columns_to_lower].apply(lambda x: x.str.lower())

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
    claude_df = pd.read_csv(match_file, sep="\t", quotechar='"')
    
    claude_df.columns = ["input_desc", "match_claude"]

    # Remove pesky quotes
    inputdf = inputdf.map(lambda x: re.sub(r'[\'"]', '', str(x)))
    targetdf = targetdf.map(lambda x: re.sub(r'[\'"]', '', str(x)))
    truthdf = truthdf.map(lambda x: re.sub(r'[\'"]', '', str(x)))
    claude_df = claude_df.map(lambda x: re.sub(r'[\'"]', '', str(x)))
    print()

    print("Length of claude table", len(claude_df))

    # check that every ground truth input was also evaluated/output by Claude
    claude_inputs_not_in_ground_truth = []
    for item in inputdf:
        if item not in claude_df["input_desc"]:
            claude_inputs_not_in_ground_truth.append(item)
    
    print()
    print("If this next part is 0, we did not hallucinate any inputs")
    print("Number of claude inputs not in ground truth:", len(claude_inputs_not_in_ground_truth))
    print()

    # merge the ground truth and the predictions into one data frame, linking on the input description
    merged_df = pd.merge(claude_df, truthdf, on="input_desc", how="left")
    
    # check
    merged_df.to_csv("tmp_predictions.csv", index=False, sep="\t")

    is_match = []
    for i in range(0, len(merged_df)):
        if (merged_df.iloc[i]["match_claude"] == merged_df.iloc[i]["target_desc"]):
            is_match.append(1)
        else:
            is_match.append(0)

    print()
    print("length is ", len(merged_df))
    print("merged_df.iloc[1][target_desc] is ", merged_df.iloc[1]["target_desc"] )
    print("merged_df.iloc[1][match_claude] is ", merged_df.iloc[1]["match_claude"])
    print("is_match[1]", is_match[1])
    print()  
    print("merged_df.iloc[2][target_desc] is ", merged_df.iloc[2]["target_desc"] )
    print("merged_df.iloc[2][match_claude] is ", merged_df.iloc[2]["match_claude"])
    print("is_match[2]", is_match[2])
    print()  
    print("merged_df.iloc[3][target_desc] is ", merged_df.iloc[3]["target_desc"] )
    print("merged_df.iloc[3][match_claude] is ", merged_df.iloc[3]["match_claude"])
    print("is_match[3]", is_match[3])
    print()  
    print("merged_df.iloc[4][target_desc] is ", merged_df.iloc[4]["target_desc"] )
    print("merged_df.iloc[4][match_claude] is ", merged_df.iloc[4]["match_claude"])
    print("is_match[4]", is_match[4])
    print()  
    print("merged_df.iloc[5][target_desc] is ", merged_df.iloc[5]["target_desc"] )
    print("merged_df.iloc[5][match_claude] is ", merged_df.iloc[5]["match_claude"])
    print("is_match[5]", is_match[5])
    print()  

    # Check data types
    #print("Column dtypes:")
    #print(merged_df[['label', 'target_desc', 'match_claude']].dtypes)

    # Check for string vs integer labels
    #print("Unique labels:", merged_df['label'].unique())
    #print("Label types:", [type(x) for x in merged_df['label'].unique()])

    merged_df["is_match"] = is_match
    
    print("Accuracy from first 100 matches:", merged_df["is_match"][:100].sum() / 100)
    print("Accuracy (whole dataset)", merged_df["is_match"].sum() / len(merged_df))

    print()

    # output mismatches
    debug_file = f"{prefix}_pred_is_wrong.csv"
    merged_df[merged_df["is_match"] == 0].to_csv(debug_file, index=False)
    # output all predictions
    all_pred_file = f"{prefix}_predictions_merged.csv"
    merged_df.to_csv(all_pred_file, index=False)