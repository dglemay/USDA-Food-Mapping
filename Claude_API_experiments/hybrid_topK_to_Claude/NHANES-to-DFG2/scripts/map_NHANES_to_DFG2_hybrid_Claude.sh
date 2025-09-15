#pip install -U sentence-transformers

## topK accuracy with semantic mapping
python3 -u semantic_topK_accuracy.py ../data/raw/input_desc_list.csv ../data/raw/target_desc_list.csv ../data/raw/nhanes_dfg2_labels.csv > ../results/output_NHANESexp4_topK_accuracy.txt

## hybrid approach of providing top 5 matches with highest similarity scores as a prompt to Claude Haiku
python3 -u hybrid_semantic_haiku2.py ../data/raw/input_desc_list.csv ../data/raw/target_desc_list.csv 5 ../results/hybrid_haiku2_NHANESexp4.txt
python checking_nhanes_exp4_claude_api.py ../data/raw/input_desc_list.csv ../data/raw/target_desc_list.csv ../data/raw/nhanes_dfg2_labels.csv ../results/hybrid_haiku2_NHANESexp4.txt ../results/chk_hybrid_haiku2_NHANESexp4

## hybrid approach of providing top 5 matches with highest similarity scores as a prompt to Claude Sonnet
python3 -u hybrid_semantic_sonnet.py ../data/raw/input_desc_list.csv ../data/raw/target_desc_list.csv 5 ../results/hybrid_sonnet_NHANESexp4.txt
python checking_nhanes_exp4_claude_api.py ../data/raw/input_desc_list.csv ../data/raw/target_desc_list.csv ../data/raw/nhanes_dfg2_labels.csv ../results/hybrid_sonnet_NHANESexp4.txt ../results/chk_hybrid_sonnet_NHANESexp4
