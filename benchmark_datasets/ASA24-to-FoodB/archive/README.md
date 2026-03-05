## This archival space provides the benchmark ASA24-to-FoodB dataset that was used in preliminary experiments.

See https://github.com/dglemay/USDA-Food-Mapping/tree/main/benchmark_datasets/ASA24-to-FoodB/dataset for final benchmark (ground truth) data that should be used for all experiments going forward. 

*ASA24_FooDB_codematches_6-26-2025.xlsx* provides matches of the ASA24 data to the FooDB database. <br>
The columns of ASA24_FooDB_codematches_6-26-2025.xlsx are described in the Dictionary tab.  <br>

*FooDB_Unique_Descriptions.csv* provides all food text descriptions of the target database, whether or not they match. <br>
Columns:  <br>
> food_id: Unique identifier for broad-level food in FooDB <br>
> food_name: Broad-level food description in FooDB <br>
> orig_food_id: Unique food identifier from source or database in citation column <br>
> orig_food_common_name_uncleaned: Fine-level description of the food from source or database in citation column, has not undergone text cleaning  <br>
> orig_food_common_name: Fine-level description of the food from source or database in citation column, has undergone text cleaning  <br>
> citation: original database or publication (FoodB includes data from other databases)  <br>
> food_V2_ID:  In-house unique identifier for fine-level description of foods <br>
