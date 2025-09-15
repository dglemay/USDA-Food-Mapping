#### Parse ASA24_to_FooDB databases
python parse_ASA24_to_FooDB.py

#### Try Haiku on the 1toMany and no ID match cases
python choose_from_1toMany_Claude.py ../data/processed/input_1toMany_ASA24toFooDB.txt ../data/processed/target_desc_FooDB.txt Haiku ../results/matched_1toMany_Haiku.txt
python hybrid_semantic_Claude.py ../data/processed/input_desc_ASA24toFooDB.txt ../data/processed/target_desc_FooDB.txt 10 Haiku ../results/matched_FooDB_hybrid_Haiku_k10.txt

#### Check accuracy
python clean_text_groundtruth.py

python checkingcsv_asa24tofoodb.py ../data/processed/input_1toMany_ASA24toFooDB.txt ../data/processed/target_desc_FooDB.txt ../data/processed/groundtruth_ASA24toFooDB.txt ../results/matched_1toMany_Haiku.txt ../results/haiku_1toMany

python checkingcsv_asa24tofoodb.py ../data/processed/input_desc_ASA24toFooDB.txt ../data/processed/target_desc_FooDB.txt ../data/processed/groundtruth_ASA24toFooDB.txt ../results/matched_FooDB_hybrid_Haiku_k10.txt ../results/haiku_ASA24toFoodDB_desc

#### Try Sonnet on the 1toMany and no ID match cases
python choose_from_1toMany_Claude.py ../data/processed/input_1toMany_ASA24toFooDB.txt ../data/processed/target_desc_FooDB.txt Sonnet ../results/matched_1toMany_Sonnet.txt
python hybrid_semantic_Claude.py ../data/processed/input_desc_ASA24toFooDB.txt ../data/processed/target_desc_FooDB.txt 10 Sonnet ../results/matched_FooDB_hybrid_Sonnet_k10.txt

#### Check accuracy
python checkingcsv_asa24tofoodb.py ../data/processed/input_1toMany_ASA24toFooDB.txt ../data/processed/target_desc_FooDB.txt ../data/processed/groundtruth_ASA24toFooDB.txt ../results/matched_1toMany_Sonnet.txt ../results/sonnet_1toMany

python checkingcsv_asa24tofoodb.py ../data/processed/input_desc_ASA24toFooDB.txt ../data/processed/target_desc_FooDB.txt ../data/processed/groundtruth_ASA24toFooDB.txt ../results/matched_FooDB_hybrid_Sonnet_k10.txt ../results/sonnet_ASA24toFoodDB_desc

#### retry the hybrid experiment with k=5
python hybrid_semantic_Claude.py ../data/processed/input_desc_ASA24toFooDB.txt ../data/processed/target_desc_FooDB.txt 5 Haiku ../results/matched_FooDB_hybrid_Haiku_k5.txt
python hybrid_semantic_Claude.py ../data/processed/input_desc_ASA24toFooDB.txt ../data/processed/target_desc_FooDB.txt 5 Sonnet ../results/matched_FooDB_hybrid_Sonnet_k5.txt

python checkingcsv_asa24tofoodb.py ../data/processed/input_desc_ASA24toFooDB.txt ../data/processed/target_desc_FooDB.txt ../data/processed/groundtruth_ASA24toFooDB.txt ../results/matched_FooDB_hybrid_Haiku_k5.txt ../results/haiku_k5_ASA24toFoodDB_desc
python checkingcsv_asa24tofoodb.py ../data/processed/input_desc_ASA24toFooDB.txt ../data/processed/target_desc_FooDB.txt ../data/processed/groundtruth_ASA24toFooDB.txt ../results/matched_FooDB_hybrid_Sonnet_k5.txt ../results/sonnet_k5_ASA24toFoodDB_desc

# Stitch altogether 
python map_ASA24_to_FooDB.py 

# let's make a version of the mapping file with no header so that we can use the same checker script
tail -n +2 ../results/complete_map_ASA24toFooDB.txt > ../results/no_header_complete_map_ASA24toFooDB.txt

# Check accuracy
python checkingcsv_asa24tofoodb.py ../data/processed/input_clean_ASA24toFooDB.txt ../data/processed/target_desc_FooDB.txt ../data/processed/groundtruth_ASA24toFooDB.txt ../results/no_header_complete_map_ASA24toFooDB.txt ../results/chk_complete_ASA24toFooDB
