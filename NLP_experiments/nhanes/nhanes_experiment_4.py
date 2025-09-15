from util import load_nhanes, clean_text, compute_accuracy_with_tresholding
from matching_algorithms.fuzzy_match import fuzzy_match
from matching_algorithms.tfidf_match import tfidf_match
from matching_algorithms.embed_match import embed_match
import pandas as pd

def nhanes_experiment_4_run():
    df = load_nhanes()
    df["index"] = [i for i in range(len(df))]

    input_desc_list = df["input_desc"].to_list()

    target_desc_list = pd.read_csv("data/dfg2_food_descriptions.csv")["simple_name"]
    target_desc_list = [s.lower() for s in target_desc_list]
    target_desc_list = list(set(target_desc_list))

    input_desc_clean_list = clean_text(input_desc_list)
    target_desc_clean_list = clean_text(target_desc_list)

    """
        After cleaning, some distinct target descriptions may become identical.
        For example, two different raw targets might clean to the same string.
        
        If we don’t track the original (pre-cleaned) target descriptions, 
        this can cause inflated accuracy scores due to multiple cleaned targets
        mapping to the same cleaned string.

        To avoid this, we create a mapping from each unique cleaned target back
        to its original raw form.
    """ 
    clean_to_raw_target_dict = dict()
    for raw, clean in zip(target_desc_list, target_desc_clean_list):
        if clean not in clean_to_raw_target_dict:
            clean_to_raw_target_dict[clean] = raw

    # and then we want only unique values
    target_desc_clean_list = list(set(target_desc_clean_list))

    # matching algorithm stuff happens here
    df_fuzzy = fuzzy_match(input_desc_clean_list, target_desc_clean_list)
    df_fuzzy["match_fuzzy"] = df_fuzzy["match_fuzzy"].map(clean_to_raw_target_dict)

    df_tfidf = tfidf_match(input_desc_clean_list, target_desc_clean_list)
    df_tfidf["match_tfidf"] = df_tfidf["match_tfidf"].map(clean_to_raw_target_dict)

    df_embed = embed_match(input_desc_list, target_desc_list)

    df = df.join(df_fuzzy, on="index", how="left")
    df = df.join(df_tfidf, on="index", how="left")
    df = df.join(df_embed, on="index", how="left")

    # results
    threshold = 0.95
    acc_fuzzy = compute_accuracy_with_tresholding(df, "fuzzy", threshold)
    acc_tfidf = compute_accuracy_with_tresholding(df, "tfidf", threshold)
    acc_embed = compute_accuracy_with_tresholding(df, "embed", threshold)

    # saving the results
    df_accuracy = pd.DataFrame({
        "method": ["fuzzy", "tfidf", "embed"],
        "accuracy": [acc_fuzzy, acc_tfidf, acc_embed]
    })

    df_accuracy.to_csv("results/accuracy_tables/nhanes_experiment_4_accuracy.csv", index=False)
    df.to_csv("results/csv_files/nhanes_experiment_4.csv", index=False)