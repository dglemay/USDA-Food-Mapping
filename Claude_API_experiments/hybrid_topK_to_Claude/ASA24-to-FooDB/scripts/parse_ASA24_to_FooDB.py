##################################################################################################
# parse_ASA24_to_FooDB.py
# The Purpose: Given input food descriptions/IDs from ASA24 data and the target database of 
# food descriptons/IDs (FooDB), the goal is to determine which input food IDs fall into the following 
# categories:
#  (A) input food IDs to target db IDs match 1-to-1 between the input and target databases. 
#  (B) input food IDs to target db IDs are 1-to-many
#  (C) there are no food ID matches between the too database
# 
# The script outputs four files to ../data/processed:
#  input_1to1_ASA24toFooDB.txt : contains the foodIDs and descriptions that match 1-to-1 with target db
#  input_1toMany_ASA24toFooDB.txt: contains food ID and food description only for IDs that match 1-to-many
#  input_desc_ASA24toFooDB.txt: contains food descriptions only for those that don't match by ID
#  target_ID_desc.txt : contains target ID and food description from target database
#
# Example usage: 
# python parse_ASA24_to_FooDB.py 
# 
# Author: Danielle Lemay with debug/syntax assists from Claude Sonnet. 
# Written for readability/documentation, not optimized for speed.
#################################################################################################
import pandas as pd
import csv

try:
    # Load input (ASA24 foods to match) 
    print(f"Loading input database")
    df_asa = pd.read_excel("../data/raw/ASA24_FooDB_codematches_8-26-2025.xlsx", sheet_name=0)
    # keep only columns of Input Food DB
    df_asa = df_asa[["Ingredient_code", "Ingredient_description_uncleaned"]]
    # rename columns
    df_asa.columns = ["input_id", "input_desc"]
    # turning the food descriptions to lowercase
    df_asa["input_desc"] = df_asa["input_desc"].str.lower()
    # make sure IDs are integers
    df_asa["input_id"] = df_asa["input_id"].astype(int)
    # remove row if input ID or description is NA
    df_asa = df_asa.dropna(subset=["input_id", "input_desc"])
    # if the input food ID exists twice, it should map to the same target - no reason to keep this
    input_df = df_asa.drop_duplicates(subset="input_id", keep="first")

    # Derive target from Full FooDB Database
    print(f"Loading full target database")
    targetdf = pd.read_csv("../data/raw/FooDB_Unique_Descriptions.csv", sep=",", header=0, encoding="latin-1")
    print(f"length of targetdf is {len(targetdf)}")
    targetdf = targetdf[["orig_food_common_name_uncleaned", "orig_food_id"]]
    # convert latin 1 to utf-8
    targetdf["orig_food_common_name_uncleaned"] = [s.encode("utf-8").decode("utf-8") for s in targetdf["orig_food_common_name_uncleaned"]]
    targetdf.columns = ["target_desc", "target_id"]
    # convert food description to lowercase
    targetdf['target_desc'] = targetdf['target_desc'].str.lower()
    # convert IDs to integers (there are a few non-integers, but all input_ids are integers so we already know those are not intended matches)
    targetdf['target_id'] = pd.to_numeric(targetdf['target_id'], errors='coerce')
    targetdf['target_id'] = targetdf['target_id'].astype('Int64')
    print(f"length of targetdf is {len(targetdf)}")

    # reduce the input list by those with identical ID matches with target
    input_ids = input_df['input_id'].tolist()
    print(f"the number of input IDs in source database is {len(input_ids)}")
    print(f"the number of unique IDs in source database is {len(set(input_ids))}")
    target_ids_with_NAs = targetdf['target_id'].tolist()
    target_ids = [x for x in target_ids_with_NAs if not pd.isna(x)]
    print(f"the number of input IDs in target database is {len(target_ids)}")
    print(f"the number of unique IDs in target database is {len(set(target_ids))}")

    # get the unique and non_unique of the target_ids
    seen_targetid = set()
    duplicates_targetid = set()
    for id in target_ids:
        if id in seen_targetid:
            duplicates_targetid.add(id)
        else:
            seen_targetid.add(id)

    print(f"The number of target IDs with duplicates: {len(duplicates_targetid)}") #486
    print(f"The number of target IDs that are unique: {len(seen_targetid)}") #8151

    # if an id from the food database ID has one and only one ID match in the target database, that's a match
    # print the match...keep track of IDs for which there is a duplicate (or more) in the target database 
    # or for which there is no matching ID in the target database
    input_has_no_target_id = set()
    input_has_dup_target_id = set()
    input_has_uniq_target_id = set()
    int_seen_targetids = [int(s) for s in seen_targetid]
    int_duplicates_targetids = [int(s) for s in duplicates_targetid]
    for idn in input_ids:
        if idn in int_duplicates_targetids:
            input_has_dup_target_id.add(idn)
        elif idn in int_seen_targetids:
            input_has_uniq_target_id.add(idn)
        else:
            input_has_no_target_id.add(idn)

    print(f"Number of 1-to-1 cases: {len(input_has_uniq_target_id)}")
    print(f"Number of 1-to-many cases: {len(input_has_dup_target_id)}")
    print(f"Number of no match-by-ID cases: {len(input_has_no_target_id)}")

    # let's check how many perfect text matches exist between input and target databases
    input_descs = input_df['input_desc'].tolist() 
    num_matching_desc = targetdf['target_desc'].isin(input_descs).sum()
    x_perc = (num_matching_desc/len(input_ids))*100
    print(f"Number of perfect matches of food text descriptions: {num_matching_desc} or {x_perc}%")

    # Write the cleaned input file : contains all food ID and food descriptions
    input_df.to_csv('../data/processed/input_clean_ASA24toFooDB.txt', sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)

    # Write input_1to1_ASA24toFooDB.txt : contains the food ID and food description that match 1-to-1 with target db
    input_df_1to1 = input_df[input_df["input_id"].isin(input_has_uniq_target_id)]
    input_df_1to1.to_csv('../data/processed/input_1to1_ASA24toFooDB.txt', sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)

    # Write input_1toMany_ASA24toFooDB.txt: contains food ID and food description only for IDs that match 1-to-many
    input_df_1toMany = input_df[input_df["input_id"].isin(input_has_dup_target_id)]
    input_df_1toMany.to_csv('../data/processed/input_1toMany_ASA24toFooDB.txt', sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)

    # Write input_desc_ASA24toFooDB.txt: contains food descriptions only for those that don't match by ID
    input_df_no_targetID = input_df[input_df["input_id"].isin(input_has_no_target_id)]
    input_df_no_targetID.to_csv('../data/processed/input_desc_ASA24toFooDB.txt', sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)

    # Write text-cleaned target database 
    targetdf.to_csv('../data/processed/target_desc_FooDB.txt', sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)

    print(f"\nDone! Files are in the ../data/processed folder.")

except Exception as e:
    # handle any exception and access the exception details
    print(f"An error occurred: {e}")