##############################################################################################
# single_NHANES_match_ClaudeAPI.py                           
# Purpose: match each Food Description from the AMPM output of NHANES 
# (vocabulary = FNDDS database) to a different target database.
# The target database is sufficiently small that it is sent with each API call.
# This version uses Sonnet 4 model: claude-sonnet-4-20250514.
#
# Inputs: list_a.txt, list_b.txt
# Output: file with two columns, item from List A, best Match from List B
#
# Author: Claude Sonnet 4 & D.Lemay
# Date: 8/10/2025
################################################################################################

import anthropic
import os
import time
import csv
from typing import List, Optional

# Configuration
API_KEY = os.getenv('ANTHROPIC_API_KEY')
MODEL = "claude-sonnet-4-20250514"  # Claude Sonnet 4
OUTPUT_FILE = "matched_NHANES_exp4_results.txt"
RATE_LIMIT_DELAY = 1  # seconds between API calls to avoid rate limiting

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=API_KEY)

def find_best_match(item_a: str, list_b: List[str]) -> str:
    """
    Find the best match for item_a in list_b using Claude Sonnet 4 API.
    Returns the best match or "none" if no good match is found.
    """
    try:
        # Create prompt for Claude
        list_b_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(list_b)])
        
        prompt = f"""Given the following item: "{item_a}"

Find the best match from this list:
{list_b_str}

Rules:
- Return only the exact text of the best matching item from the list above
- If no item is a reasonably good match (semantic similarity < 70%), return exactly "none"
- Consider semantic similarity, synonyms, nutrient composition, and conceptual relationships
- Be selective - only return matches that are genuinely similar in meaning
- Your response should contain ONLY the matching text or "none", nothing else

Best match:"""

        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            temperature=0.1,
            system="You are a precise text matching assistant. Return the best matching text from the provided list or 'none'. Do not include explanations, quotes, or additional text.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.content[0].text.strip().strip('"\'')
        
        # Validate that the result is either "none" or actually in list_b
        if result.lower() == "none":
            return "none"
        elif result in list_b:
            return result
        else:
            # If Claude returned something not exactly in list_b, try to find closest match
            result_lower = result.lower()
            for item in list_b:
                if item.lower() == result_lower:
                    return item
                elif item.lower() in result_lower or result_lower in item.lower():
                    return item
            return "none"
            
    except Exception as e:
        print(f"Error processing '{item_a}': {e}")
        return "none"

def match_lists(list_a: List[str], list_b: List[str], output_file: str):
    """
    Match items from list_a to list_b and save results to a tab-delimited file.
    """
    results = []
    
    print(f"Matching {len(list_a)} items from List A to {len(list_b)} items in List B...")
    
    for i, item_a in enumerate(list_a, 1):
        print(f"Processing item {i}/{len(list_a)}: {item_a}")
        
        # Find best match
        match = find_best_match(item_a, list_b)
        
        # Store result
        results.append([item_a, match])
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Write results to tab-delimited file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Write header
        writer.writerow(['List_A_Item', 'Best_Match_from_List_B'])
        
        # Write data
        writer.writerows(results)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    matched_count = sum(1 for result in results if result[1] != "none")
    print(f"Summary: {matched_count}/{len(list_a)} items matched")

def load_lists_from_files(file_a: str, file_b: str) -> tuple[List[str], List[str]]:
    """
    Load lists from text files (one item per line).
    """
    with open(file_a, 'r', encoding='utf-8') as f:
        list_a = [line.strip() for line in f if line.strip()]
    
    with open(file_b, 'r', encoding='utf-8') as f:
        list_b = [line.strip() for line in f if line.strip()]
    
    return list_a, list_b

# Example usage
if __name__ == "__main__":
    # Load from files 
    list_a, list_b = load_lists_from_files('input_desc_list.txt', 'target_desc_list.txt')
    
    # Perform matching with Claude Sonnet 4
    match_lists(list_a, list_b, OUTPUT_FILE)
    
    print(f"\nDone! Check {OUTPUT_FILE} for results.")
