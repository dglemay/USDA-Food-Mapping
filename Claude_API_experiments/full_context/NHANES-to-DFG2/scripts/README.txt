Ran NHANES experiment 4 on Claude Sonnet 4 on Aug 10, 2025
using
python single_NHANES_match_ClaudeAPI.py
Input files = input_desc_list.txt, target_desc_list.txt
Output file = matched_NHANES_exp4_results.txt

Returned the best match. If no good match was found, then "none."

Of the 1304 items in the first list being matched to the 256 items in the target list, 1080 of the 1304 items were matched.
The remainders had no match ("none").

The cost of running the script (which prompts Claude 1304 times) was $8.82.

-------------------------------------------------------------------

Ran NHANES experiment 4 on Claude Haiku on Aug 14, 2025
using
python3 -u haiku_NHANES_match_ClaudeAPI.py >& stdout_haiku_NHANES.txt

MODEL = "claude-3-haiku-20240307"
Input files = input_desc_list_noquotes.txt, target_desc_list_noquotes.txt
Output file = matched_haiku_NHANES_exp4_results.txt

Of the 1304 items in the first least being matched to the 256 items in the target list, 624 were no match ("none").

The cost of running the script was less than $1. (0.85 - 0.13)
The runtime was 37 minutes.
