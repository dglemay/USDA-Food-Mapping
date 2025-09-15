##################################################################################################
# checking_nhanes_exp_4_claude_api.py
# Purpose: this code checkes the results of NHANES experiment 4 against ground truth
# 
# Inputs: input food database, target food database, ground truth, Claude results
# Output prefix to create 
#   PREFIX_stdout.txt
#   PREFIX_merged_preditions.csv
#   PREFIX_label_is_1_and_pred_is_wrong.csv
#   PREFIX_label_is_0_but_pred_match.csv
#
# Example usage: python checking_nhanes_exp_4_claude_api.py input_desc_list.csv target_desc_list.csv nhanes_dfg2_labels.csv matched_haiku_NHANES_exp4_results.txt haiku_NHANESexp4
# 
# Authors: Danielle Lemay & Michael Strohmeier
#################################################################################################
import sys
import pandas as pd
import re

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

# function to read input data, ground truth
def load_input_files(input_file, target_file, truth_file):
    """Load three CSV files and return as DataFrames"""
    try:
        print("Loading input file")
        inputdf = pd.read_csv(input_file, sep=",", header=0, dtype=str) #first row is header

        print("Loading target file")
        targetdf = pd.read_csv(target_file, sep=",", header=0, dtype=str) #first row is header

        print("Loading truth file")
        truthdf = pd.read_csv(truth_file, sep=",", header=0, dtype=str) #first row is header
        truthdf['label'] = pd.to_numeric(truthdf['label'])

        truthdf.rename(columns={
            'ingred_desc': 'input_desc',
            'ingred_code' : 'input_id',
            'simple_name': 'target_desc',
            'label': 'label' 
            }, inplace=True)

        # turning the food descriptions in each column to lowercase
        inputdf['input_desc'] = inputdf['input_desc'].str.lower()
        targetdf['target_desc'] = targetdf['target_desc'].str.lower()
        truthdf['input_desc'] = truthdf['input_desc'].str.lower()
        truthdf['target_desc'] = truthdf['target_desc'].str.lower()

        # drop duplicates, if they exist
        inputdf = inputdf.drop_duplicates(subset="input_desc", keep="first")
        targetdf = targetdf.drop_duplicates(subset="target_desc", keep="first")
        truthdf = truthdf.drop_duplicates(subset="input_desc", keep="first")
        
        print(f"Number of unique inputs {len(inputdf)}")
        print(f"Number of unique targets {len(targetdf)}") 
        print(f"Number of unique input items in ground truth table {len(truthdf)}")

        # let's check how many perfect text matches exist between input and target databases
        input_descs = inputdf['input_desc'].tolist() 
        num_matching_desc = targetdf['target_desc'].isin(input_descs).sum()
        x_perc = (num_matching_desc/len(inputdf))*100
        print(f"Number of perfect matches of food text descriptions: {num_matching_desc} or {x_perc}%")

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

    # checking original dataset
    num_0s = len(truthdf[truthdf['label'] == 0]) # ground truth number of nones
    num_1s = len(truthdf[truthdf['label'] == 1])
    print("Length of ground truth table", len(truthdf))
    print("Ground truth number of 0s (nones):", num_0s)
    print("Ground truth number of 1s (match exists):", num_1s)

    # read claude's results
    #claude_truthdf = pd.read_csv(input_file2, sep="\t", quotechar='"')
    claude_df = pd.read_csv(match_file, sep="\t", header=0, names=['input_desc', 'match_claude'])
    claude_df['input_desc'] = claude_df['input_desc'].str.lower()
    claude_df['match_claude'] = claude_df['match_claude'].str.lower()

    # Remove pesky double quotes
    truthdf = truthdf.map(lambda x: re.sub(r'[\"]', '', str(x)))
    claude_df = claude_df.map(lambda x: re.sub(r'[\"]', '', str(x)))
   
    print()

    num_pred_none = len(claude_df[claude_df["match_claude"] == "none"])
    print("Length of claude table", len(claude_df))
    print("claude number of predicted 0s (none)", num_pred_none)

    max_possible_accuracy = ((num_0s - num_pred_none + num_1s) / len(truthdf))
    print("max POSSIBLE accuracy IF every match WAS correct", max_possible_accuracy)

    # check that every ground truth input was also evaluated/output by Claude
    unique_ground_truth_inputs = truthdf["input_desc"].unique()
    
    claude_inputs_not_in_ground_truth = []
    for input_desc in claude_df["input_desc"]:
        if input_desc not in unique_ground_truth_inputs:
            claude_inputs_not_in_ground_truth.append(input_desc)
            print(f"Input description {input_desc} not in ground truth database")
    print()
    print("If this next part is 0, we did not hallucinate any inputs")
    print("Number of claude inputs not in ground truth:", len(claude_inputs_not_in_ground_truth))
    print()

    print("Check that every output from Claude actually existed in the target database")
    # check that every output from Claude actually existed in the target database 
    claude_outputs_not_in_target_database = []
    for item in claude_df['match_claude']:
        if item != 'none':
            if item not in targetdf['target_desc'].values:
                claude_outputs_not_in_target_database.append(item)
                print(f"Item {item} not in target database")
    
    print()
    print("If this next part is 0, we did not hallucinate any responses not in target database")
    print("Number of claude outputs not in target database:", len(claude_outputs_not_in_target_database))
    print()

    # merge the ground truth and the predictions into one data frame, linking on the input description
    merged_df = pd.merge(truthdf, claude_df, on="input_desc", how="left")
     # Convert the label column to integers
    merged_df['label'] = merged_df['label'].astype(int)

    # this is just checking if (match=="none" and label == 0), which is a valid match
    # OR (match==target and label == 1), which is a valid match
    # else in either case of label 0 or 1 if the  additional criteria is not met then it is not a match
    is_match = []
    for i in range(0, len(merged_df)):
        if merged_df.iloc[i]["label"] == 1:
            if merged_df.iloc[i]["target_desc"] == merged_df.iloc[i]["match_claude"]:
                is_match.append(1)
            else:
                is_match.append(0)
        elif merged_df.iloc[i]["label"] == 0:
            if ((merged_df.iloc[i]["match_claude"] == "none")):
                is_match.append(1)
            else:
                is_match.append(0)

    #print()
    #print("length is ", len(merged_df))
    #print("merged_df.iloc[0][label] is ", merged_df.iloc[0]["label"])
    #print("merged_df.iloc[0][target_desc] is ", merged_df.iloc[0]["target_desc"] )
    #print("merged_df.iloc[0][match_claude] is ", merged_df.iloc[0]["match_claude"])
    #print("is_match is", is_match)
    #print()  

    # Check data types
    #print("Column dtypes:")
    #print(merged_df[['label', 'target_desc', 'match_claude']].dtypes)

    # Check for string vs integer labels
    #print("Unique labels:", merged_df['label'].unique())
    #print("Label types:", [type(x) for x in merged_df['label'].unique()])

    merged_df["is_match"] = is_match
    
    print("Percent accuracy from first 100 matches:", merged_df["is_match"][:100].sum())
    print("Percent accuracy (whole dataset)", (merged_df["is_match"].sum() / len(merged_df))*100)

    print()
    merged_df_label_0_only = merged_df[merged_df["label"] == 0]
    print("merged_df_label_0_only number of rows: ", len(merged_df_label_0_only))
    print("Percent accuracy when only checking label == 0 ('None' is the valid match):", (merged_df_label_0_only["is_match"].sum() / len(merged_df_label_0_only))*100)

    merged_df_label_1_only = merged_df[merged_df["label"] == 1]
    print("merged_df_label_1_only number of rows: ", len(merged_df_label_1_only))
    print("Percent accuracy when only checking label == 1 (a valid match does exist):", (merged_df_label_1_only["is_match"].sum() / len(merged_df_label_1_only))*100)

    # output mismatches
    debug_file = f"{prefix}_label_is_1_pred_is_wrong.csv"
    merged_df_label_1_only[merged_df_label_1_only["is_match"] == 0].to_csv(debug_file, index=False)
    debug2_file = f"{prefix}_label_is_0_but_pred_match.csv"
    merged_df_label_0_only[merged_df_label_0_only["is_match"] == 0].to_csv(debug2_file, index=False)

    # output all predictions
    all_pred_file = f"{prefix}_predictions_merged.csv"
    merged_df.to_csv(all_pred_file, index=False)
