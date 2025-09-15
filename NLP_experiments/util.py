import os
import spacy
import re
import pandas as pd

def mkdir_results():
    # creating results directory
    results_dir = "results/"
    os.makedirs(results_dir, exist_ok=True)

    # creating sub directories for accuracy tables and csv files
    accuracy_dir = os.path.join(results_dir, "accuracy_tables")
    csv_dir      = os.path.join(results_dir, "csv_files")

    os.makedirs(accuracy_dir, exist_ok=True)
    os.makedirs(csv_dir,      exist_ok=True)

def load_asa():
    df = pd.read_excel("data/ASA24_FooDB_codematches_6-26-2025.xlsx", sheet_name=0)
    df = df[["Ingredient_description_uncleaned", "orig_food_common_name_uncleaned"]]
    df.columns = ["input_desc", "target_desc"]

    # turning the food descriptions in each column to lowercase
    columns_to_lower = ["input_desc", "target_desc"]
    df[columns_to_lower] = df[columns_to_lower].apply(lambda x: x.str.lower())

    # remove row if input or target is NA
    df = df.dropna(subset=["input_desc", "target_desc"])

    # if the input exists twice, it should map to the same target - no reason to keep this
    df = df.drop_duplicates(subset="input_desc", keep="first")

    return df

def load_nhanes():
    df = pd.read_csv("data/nhanes_dfg2_labels.csv")

    df = df[["ingred_desc", "simple_name", "label"]]
    df.columns = ["input_desc", "target_desc", "label"]

    # turning the food descriptions in each column to lowercase
    columns_to_lower = ["input_desc", "target_desc"]
    df[columns_to_lower] = df[columns_to_lower].apply(lambda x: x.str.lower())

    df = df.drop_duplicates(subset="input_desc", keep="first")

    return df

def is_valid_token(token):
    return (
        not token.is_stop and 
        not token.is_punct and 
        not token.is_space and
        token.is_alpha
    )

def clean_text(arr):
    assert isinstance(arr, (list, pd.Series)), "arr must be a list or pd.Series"

    arr = [re.sub(r"[^a-zA-Z\s]", " ", s).lower().strip() for s in arr]

    nlp = spacy.load("en_core_web_sm")

    cleaned_tokens = [" ".join([token.lemma_.lower() for token in nlp(s) if is_valid_token(token)]) for s in arr]
    return cleaned_tokens

def compute_accuracy_simple(df, method):
    # check if column exists (match method exists in our dataframe)
    col = f"match_{method}"
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in DataFrame.")
    
    # if the predicted match is the same as what is in that column then it is a match, it's a boolean mask
    correct = df[col] == df["target_desc"]
    accuracy = correct.sum() / len(df)

    return accuracy

def compute_accuracy_with_tresholding(df, method, threshold, is_llm=0):
    if not is_llm:
        correct_match = (
            ((df["label"] == 0) & (df[f"score_{method}"] < threshold))
            |
            ((df["label"] == 1) & (df[f"score_{method}"] >= threshold) & (df[f"match_{method}"] == df["target_desc"]))
        )
    else:
        # not using a similarity score to predict if description is not in database
        # instead the LLMs were meant to output "NO MATCH" in such a case
        correct_match = (
            ((df["label"] == 0) & (df[f"score_{method}"] == "NO MATCH"))
            |
            ((df["label"] == 1) & (df[f"match_{method}"] == df["target_desc"]))
        )

    accuracy = correct_match.mean()

    return accuracy