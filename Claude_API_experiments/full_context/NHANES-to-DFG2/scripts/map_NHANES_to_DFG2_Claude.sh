# Run experiments with Claude models for NHANES-to-DFG2 given full target database
#python sonnet_NHANES_match_ClaudeAPI.py
#python3 -u haiku_NHANES_match_ClaudeAPI.py >& stdout_haiku_NHANES.txt

#check results
python checking_nhanes_exp4_claude_api.py ../data/processed/input_desc_list.csv ../data/processed/target_desc_list.csv ../data/raw/nhanes_dfg2_labels.csv ../results/matched_haiku_NHANES_exp4_results.txt ../results/haiku_NHANESexp4
python checking_nhanes_exp4_claude_api.py ../data/processed/input_desc_list.csv ../data/processed/target_desc_list.csv ../data/raw/nhanes_dfg2_labels.csv ../results/matched_NHANES_exp4_results.txt ../results/sonnet_NHANESexp4

