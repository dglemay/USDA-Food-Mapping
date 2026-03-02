# USDA Food Matching

The **FoodMapper application** can be found at https://github.com/RichardStoker-USDA/FoodMapper/ 

This repository contains ground truth data, evaluation methods, and results for the paper, "Evaluation of Large Language Models for Mapping Dietary Data to Food Databases," under review. Authors = Danielle G. Lemay, Michael P. Strohmeier, Richard B. Stoker, Jules A. Larke, Stephanie M.G. Wilson

- **Ground truth dataset ASA24-to-FoodB** 

  - Input food descriptions = Foods reported via ASA24 dietary recall
  - Target database = FoodB
  - Assumes **every** input food *does* have a valid match in the target set.
  - Goal: Return the single best match for each input food.

- **Ground truth dataset  NHANES-to-DFG2** 

  - Input food descriptions = Foods reported via 24-hr dietary recall in NHANES
  - Target database = Davis Food Glycopedia 2
  - Some input descriptions may have **no suitable match**.
  - Goal: Return the best match if one exists; otherwise, flag the input as `NO MATCH`.

---

## Setup and Installation

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### 1. Clone the Repository and Change Directory
`git clone https://github.com/dglemay/USDA-Food-Mapping-main`<br>
`cd USDA-Food-Mapping-main/NLP_experiments`
### 2. Create the Conda Environment and Activate
`conda env create -f environment.yml`<br>
`conda activate USDA-food-mapping`
### 3. Download the SpaCy Language Model
`python -m spacy download en_core_web_sm`
### 4. Run the NLP Experiments
`python main.py`

This will:

1. Create a results/ directory with subdirectories for accuracy tables and CSV files
2. Run the NLP algorithms on the ASA24-to-FoodB dataset 
3. Run the NLP algorithms on the NHANES-to-DFG2 dataset
4. Save results to the results/ directory

---

## Data
- **ASA24**<br>
***File***: `ASA24_FooDB_codematches_6-26-2025.xlsx`<br>
This dataset assumes that every input description has a corresponding match in the target set.

<br>

- **NHANES**<br>
***File***: `nhanes_dfg2_labels.csv`<br>
In this dataset, each ingredient description (ingred_desc) derived from the dietary data was labeled as either having a valid match (label = 1) or not (label = 0) to the food description (simple_name) in the Davis Food Glycopedia 2.0. Matches were considered to exist if there was a good match for the exact food **or** for a food with similar carbohydrate content, which served as a proxy when no direct match was available.

**Data provenance**   > *[add citation / download URL here]*

---

## NLP Matching Algorithms (`NLP_experiments/matching_algorithms/`)

| Method         | File              | Description                                                                                                                                                                                                                   |
|----------------|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Fuzzy          | `fuzzy_match.py` | Selects the best match based on fuzzy string similarity using the RapidFuzz library. For each input, the algorithm chooses the closest target string according to `fuzz.ratio`. Text is cleaned (lowercased, punctuation removed, whitespace collapsed) before matching. |
| TF‑IDF         | `tfidf_match.py` | Selects the best match by computing cosine similarity between TF‑IDF vectors of cleaned input and target descriptions. Text is cleaned before vectorization (lowercased, punctuation removed, and whitespace collapsed).                                                                                                                              |
| Embedding      | `embed_match.py` | Selects the best match by computing cosine similarity between dense embeddings generated using `SentenceTransformer("thenlper/gte-large")`. Input and target texts are used **without** cleaning or preprocessing applied.                                                                 |                                                                |

## Matching with Claude family LLMs (Anthropic) (`Claude_API_experiments/`)

## Exploring LLM model size and Prompt Strategy with Gemma3 (`SCINet_Gemma3_experiments/`)
