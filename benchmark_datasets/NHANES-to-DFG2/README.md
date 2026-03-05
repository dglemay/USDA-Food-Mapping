## Benchmark (Ground Truth) Dataset for NHANES-to-DFG2

This is a benchmark for automated mapping based on food text descriptions between NHANES dietary data <br>
and the Davis Food Glycopedia 2.0 data set. All food descriptions in this benchmark are raw/uncleaned. <br>

This benchmark provides two files: <br>

**nhanes_dfg2_labels.csv** - contains all matches between the foods reported in NHANES and the matches <br>
in the Davis Food Glycopedia 2.   <br>
column descriptions:<br>
>   ingred_desc: FNDDS food text description from NHANES dietary data. <br>
>   ingred_code: FNDDS food code, also from NHANES dietary data. <br>
>   simple_name: food text description from DFG2. <br>
>   label: Label == 0 indicates there is No Match. Label == 1 indicates an appropriate Match.<br>

**dfg2_food_descriptions.csv** - contains the entire Davis Food Glycopedia 2 list of food text descriptions, <br>
including foods that don't match <br>
