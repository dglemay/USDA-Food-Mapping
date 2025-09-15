Ran ASA24 experiment 4 on Claude Sonnet 4 via API but ran out of credits (>$15). 
Possible strategies include:
  Send the whole target database with every API call (did not attempt due to expense)
  Cache the whole target database (not possible due to rate-limiting as db ~100,000 tokens)
  Chunk the database into < 30,000 tokens; For each chunk, cache the chunk and send just a short user prompt 
  of the food description to be matched. (this was the approach taken that resulted in >$15 without completing)
  Switched to Haiku for cheaper approach.

--------------------------------------------------------------
Ran ASA24 experiment 4 on Claude Haiku via API:

python3 -u haiku_asa24_match.py >& stdout_haiku_asa24exp4_match_081525.txt

MODEL = "claude-3-haiku-20240307"
Input files = input_desc_list_noquotes.txt, target_desc_list_noquotes.txt
Output file = matched_haiku_ASA24_exp4_081625.txt

The cost of running the script was less than $2.
The runtime was approximately 9 hours.
