##################################################################################################
# text_clean_groundtruth.py
# The Purpose: Clean up the text of the ground truth the same way we did for the input
# 
# The script outputs one file to ../data/processed:
#  ground_truth_ASA24toFoodB.txt : contains input food database ID and description, as well
#    as target database food ID and description
#
# Example usage: 
# python text_clean_groundtruth.py
# 
# Author: Danielle Lemay with debug/syntax assists from Claude Sonnet. 
# Written for readability/documentation, not optimized for speed.
#################################################################################################
import pandas as pd
import csv

try:
    # Load input (ASA24 foods to match) 
    print(f"Loading grouth truth")
    df = pd.read_excel("../data/raw/ASA24_FooDB_codematches_8-26-2025.xlsx", sheet_name=0)
    # keep only columns of Input Food DB
    df = df[["Ingredient_code", "Ingredient_description_uncleaned", "orig_food_common_name_uncleaned", "orig_food_id"]]
    # rename columns
    df.columns = ["input_id", "input_desc", "target_desc", "target_id"]
    # turning the food descriptions to lowercase
    df["input_desc"] = df["input_desc"].str.lower()
    df["target_desc"] = df["target_desc"].str.lower()
    # make sure IDs are integers
    df["input_id"] = df["input_id"].astype(int)
    # remove row if input ID or description is NA
    df = df.dropna(subset=["input_id", "input_desc"])
    # if the input food ID exists twice, it should map to the same target - no reason to keep this
    df = df.drop_duplicates(subset="input_id", keep="first")
    
    # convert target IDs to integers (there are a few non-integers, but all input_ids are integers so we already know those are not intended matches)
    df['target_id'] = pd.to_numeric(df['target_id'], errors='coerce')
    df['target_id'] = df['target_id'].astype('Int64')

    # Write text-cleaned target database 
    df.to_csv('../data/processed/groundtruth_ASA24toFooDB.txt', sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)

    print(f"\nDone! Files are in the ../data/processed folder.")

except Exception as e:
    # handle any exception and access the exception details
    print(f"An error occurred: {e}")