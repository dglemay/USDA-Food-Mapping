## Benchmark (Ground Truth) Dataset for ASA24-to-FoodB

This is a benchmark for automated mapping based on IDs and food text descriptions between ASA24 dietary data <br>
and the FoodB database. All food descriptions in this benchmark are raw/uncleaned. <br>

This benchmark provides two files: <br>

**groundtruth_ASA24toFooDB.txt** - contains all matches between the foods reported via ASA24 and 
the matches in the FoodB database. <br>
column descriptions:<br>
>  input_id:	ASA24 food ID <br> 
>  input_desc: ASA24 food text description <br>
>  target_desc: FooDB food text description <br>
>  target_id: FooDB food identifier <br>

**target_desc_FooDB.txt** - contains the entire FoodB list of food text descriptions, including <br>
foods that don't match <br>
column descriptions: <br>
>  target_desc: FooDB food text description <br>
>  target_id: FooDB food identifier <br>

*NOTES:* 
1) Entries in groundtruth_ASA24toFooDB.txt should be matched by ID first. Only those with mismatching IDs should be used for text-only matching exercises. 
2) To fully reproduce the NLP_experiments, please see datasets in https://github.com/dglemay/USDA-Food-Mapping/tree/main/benchmark_datasets/ASA24-to-FoodB/archive

