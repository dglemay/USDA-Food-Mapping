##############################################################################################
# haiku_ASA24_match.py                           
# Purpose: match each Food Description from an ASA24 output to a target database.
# In this version, the target database (list_b) is moved into a system prompt 
# that gets cached, meaning that it is sent once and re-used for subsequent 
# requests. API call restructured so that the user message just contains the
# item to match. Added rate limit management, token-aware batching (chunking).
# This version reprompts with the best match from each chunk of the database 
# if batching was required. This version uses HAIKU.
# Also fixes a bug that makes combining results fail.
# Added checkpoint functionality to save progress incrementally and resume.
# Inputs: list_a.txt, list_b.txt
# Output: file with two columns, item from List A, best Match from List B
#
# Author: Claude Sonnet 4 & D.Lemay
# Date: 8/12/2025, revised 8/15/2025
################################################################################################
import anthropic
import os
import time
import csv
import math
import json
import glob
from typing import List, Optional, Dict, Any

# Configuration
API_KEY = os.getenv('ANTHROPIC_API_KEY')
MODEL = "claude-3-haiku-20240307"  # Claude Haiku
OUTPUT_FILE = "matched_haiku_ASA24_exp4_081625.txt"
CHECKPOINT_DIR = "checkpoints"
RATE_LIMIT_DELAY = 0.1  # Delay between requests
TOKEN_LIMIT_PER_MINUTE = 25000  # Conservative limit (below 30k to leave buffer)
ESTIMATED_TOKENS_PER_WORD = 1.3  # Conservative estimate for token counting

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=API_KEY)

def create_checkpoint_dir():
    """Create checkpoint directory if it doesn't exist."""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"Created checkpoint directory: {CHECKPOINT_DIR}")

def save_checkpoint(chunk_idx: int, chunk_results: List[List[str]], metadata: Dict[str, Any]):
    """Save checkpoint for a completed chunk."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"chunk_{chunk_idx}_checkpoint.json")
    checkpoint_data = {
        "chunk_idx": chunk_idx,
        "results": chunk_results,
        "metadata": metadata,
        "timestamp": time.time()
    }
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    print(f"Checkpoint saved: {checkpoint_file}")

def load_checkpoint(chunk_idx: int) -> Optional[Dict[str, Any]]:
    """Load checkpoint for a specific chunk if it exists."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"chunk_{chunk_idx}_checkpoint.json")
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_file}: {e}")
            return None
    return None

def find_existing_checkpoints() -> List[int]:
    """Find all existing checkpoint files and return chunk indices."""
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "chunk_*_checkpoint.json"))
    chunk_indices = []
    
    for file_path in checkpoint_files:
        try:
            filename = os.path.basename(file_path)
            # Extract chunk index from filename like "chunk_1_checkpoint.json"
            chunk_idx = int(filename.split('_')[1])
            chunk_indices.append(chunk_idx)
        except (ValueError, IndexError):
            print(f"Warning: Couldn't parse chunk index from {file_path}")
    
    return sorted(chunk_indices)

def save_progress_state(list_a: List[str], list_b: List[str], list_b_chunks: List[List[str]], 
                       completed_chunks: List[int], output_file: str):
    """Save the overall progress state."""
    state_file = os.path.join(CHECKPOINT_DIR, "progress_state.json")
    state_data = {
        "list_a_count": len(list_a),
        "list_b_count": len(list_b),
        "total_chunks": len(list_b_chunks),
        "completed_chunks": completed_chunks,
        "output_file": output_file,
        "timestamp": time.time()
    }
    
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state_data, f, ensure_ascii=False, indent=2)

def load_progress_state() -> Optional[Dict[str, Any]]:
    """Load the overall progress state."""
    state_file = os.path.join(CHECKPOINT_DIR, "progress_state.json")
    
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading progress state: {e}")
            return None
    return None

def clean_checkpoints():
    """Clean up checkpoint files (call after successful completion)."""
    if os.path.exists(CHECKPOINT_DIR):
        checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.json"))
        for file_path in checkpoint_files:
            try:
                os.remove(file_path)
                print(f"Removed checkpoint: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text (rough approximation).
    """
    return int(len(text.split()) * ESTIMATED_TOKENS_PER_WORD)

def create_cached_system_prompt(list_b: List[str]) -> str:
    """
    Create a system prompt that includes the target list for caching.
    This will be cached across all API calls.
    """
    list_b_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(list_b)])
    
    prompt = f"""You are a precise text matching assistant. Your task is to find the best semantic match for a given item from the following target list:

{list_b_str}

Instructions:
- Return only the exact text of the best matching item from the list above
- If no item is a reasonably good match (semantic similarity < 70%), return exactly "none"
- Consider semantic similarity, synonyms, nutrient composition, and conceptual relationships
- Be selective - only return matches that are genuinely similar in meaning
- Your response should contain ONLY the matching text or "none", nothing else"""
    
    return prompt

def wait_for_rate_limit(estimated_tokens: int, last_request_time: float):
    """
    Wait if necessary to avoid rate limit based on estimated tokens.
    """
    if estimated_tokens > TOKEN_LIMIT_PER_MINUTE:
        print(f"Large request ({estimated_tokens} estimated tokens). Waiting 65 seconds for rate limit reset...")
        time.sleep(65)
        return time.time()
    
    # Calculate time since last request
    time_since_last = time.time() - last_request_time
    
    # If we're making requests too quickly for the token limit, wait
    min_interval = (estimated_tokens / TOKEN_LIMIT_PER_MINUTE) * 60
    if time_since_last < min_interval:
        wait_time = min_interval - time_since_last + 1  # +1 second buffer
        print(f"Rate limiting: waiting {wait_time:.1f} seconds...")
        time.sleep(wait_time)
    
    return time.time()

def find_best_match_cached(item_a: str, cached_system_prompt: str) -> str:
    """
    Find the best match for item_a using the cached system prompt.
    Returns the best match or "none" if no good match is found.
    """
    max_retries = 5 
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Simple user prompt - the list is already in the cached system prompt
            user_prompt = f'Find the best match for: "{item_a}"'

            response = client.messages.create(
                model=MODEL,
                max_tokens=100,
                temperature=0.1,
                system=[
                    {
                        "type": "text",
                        "text": cached_system_prompt,
                        "cache_control": {"type": "ephemeral"}  # Enable caching for this system prompt
                    }
                ],
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            result = response.content[0].text.strip().strip('"\'')
            return result
                
        except anthropic.RateLimitError as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Rate limit exceeded for '{item_a}' after {max_retries} attempts: {e}")
                return "none"
            
            wait_time = 30 * retry_count  # Exponential backoff
            print(f"Rate limit hit for '{item_a}'. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"Error processing '{item_a}': {e}")
            return "none"
    
    return "none"

def validate_match(result: str, list_b: List[str]) -> str:
    """
    Validate that the result is either "none" or actually in list_b.
    """
    if result.lower() == "none":
        return "none"
    elif result in list_b:
        return result
    else:
        # If Claude returned something not exactly in list_b, try to find closest match
        # by checking for case-insensitive match or substring (partial) match
        # A partial match is considered valid if either candidate is a substring of the other
        result_lower = result.lower()
        for item in list_b:
            if item.lower() == result_lower:
                return item
            elif item.lower() in result_lower or result_lower in item.lower():
                return item
        return "none"

def chunk_list_b(list_b: List[str], max_tokens_per_chunk: int = 20000) -> List[List[str]]:
    """
    Split list_b into smaller chunks to avoid token limits.
    """
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    base_prompt_tokens = estimate_tokens("""You are a precise text matching assistant. Your task is to find the best semantic match for a given item from the following target list:

Instructions:
- Return only the exact text of the best matching item from the list above
- If no item is a reasonably good match (semantic similarity < 70%), return exactly "none"
- Consider semantic similarity, synonyms, nutrient composition, and conceptual relationships
- Be selective - only return matches that are genuinely similar in meaning
- Your response should contain ONLY the matching text or "none", nothing else""")
    
    # Estimate tokens for the base prompt and any items collected so far
    # If adding an item would exceed the limit, start a new chunk.
    # A new chunk starts with the next item.
    for item in list_b:
        item_tokens = estimate_tokens(f"{len(current_chunk)+1}. {item}\n")
        
        if current_tokens + item_tokens + base_prompt_tokens > max_tokens_per_chunk and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [item]
            current_tokens = item_tokens
        else:
            current_chunk.append(item)
            current_tokens += item_tokens
    
    # Add the last chunk if it has items
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def match_lists_with_caching(list_a: List[str], list_b: List[str], output_file: str):
    """
    Match items from list_a to list_b using prompt caching and save results to a tab-delimited file.
    This function handles large lists by chunking list_b if necessary based on token limits.
    Each item in list_a is matched against the cached system prompt containing the chunk of list_b.
    This is repeated for each chunk of list_b if it exceeds the token limit.
    Includes checkpoint functionality to resume interrupted processing.
    """
    create_checkpoint_dir()
    
    print(f"Target list has {len(list_b)} items")
    estimated_total_tokens = estimate_tokens(create_cached_system_prompt(list_b))
    print(f"Estimated system prompt tokens: {estimated_total_tokens}")
    
    # Check if we need to chunk the list
    if estimated_total_tokens > TOKEN_LIMIT_PER_MINUTE:
        print(f"System prompt too large ({estimated_total_tokens} tokens). Chunking list_b...")
        list_b_chunks = chunk_list_b(list_b, max_tokens_per_chunk=20000)
        print(f"Split into {len(list_b_chunks)} chunks")
        
        # Check for existing progress
        progress_state = load_progress_state()
        existing_chunks = find_existing_checkpoints()
        
        if progress_state and existing_chunks:
            print(f"\nFound existing progress:")
            print(f"- Total chunks: {progress_state.get('total_chunks', 'unknown')}")
            print(f"- Completed chunks: {len(existing_chunks)} ({existing_chunks})")
            
            response = input("Do you want to resume from existing checkpoints? (y/n): ").lower().strip()
            if response == 'y':
                print("Resuming from existing checkpoints...")
            else:
                print("Starting fresh (existing checkpoints will be ignored)...")
                existing_chunks = []
        
        # Process each chunk separately and combine results
        all_results = []
        completed_chunks = []
        
        for chunk_idx, chunk in enumerate(list_b_chunks, 1):
            # Check if this chunk was already completed
            if chunk_idx in existing_chunks:
                print(f"\nChunk {chunk_idx}/{len(list_b_chunks)}: Loading from checkpoint...")
                checkpoint_data = load_checkpoint(chunk_idx)
                if checkpoint_data:
                    chunk_results = checkpoint_data['results']
                    all_results.append(chunk_results)
                    completed_chunks.append(chunk_idx)
                    print(f"Chunk {chunk_idx}: Loaded {len(chunk_results)} results from checkpoint")
                    continue
            
            print(f"\nProcessing chunk {chunk_idx}/{len(list_b_chunks)} ({len(chunk)} items)...")
            chunk_results = process_chunk(list_a, chunk, chunk_idx)
            all_results.append(chunk_results)
            completed_chunks.append(chunk_idx)
            
            # Save checkpoint for this chunk
            metadata = {
                "chunk_size": len(chunk),
                "total_chunks": len(list_b_chunks),
                "list_a_size": len(list_a)
            }
            save_checkpoint(chunk_idx, chunk_results, metadata)
            
            # Save overall progress state
            save_progress_state(list_a, list_b, list_b_chunks, completed_chunks, output_file)
        
        # Combine results - for each item in list_a, take the best match across all chunks
        print(f"\nCombining results from {len(all_results)} chunks...")
        final_results = combine_chunk_results(list_a, all_results, client)
        
    else:
        # Original single-chunk processing
        print("System prompt size acceptable. Processing as single chunk...")
        
        # Check for existing single chunk checkpoint
        existing_chunks = find_existing_checkpoints()
        if 1 in existing_chunks:
            response = input("Found existing checkpoint for single chunk. Resume? (y/n): ").lower().strip()
            if response == 'y':
                checkpoint_data = load_checkpoint(1)
                if checkpoint_data:
                    final_results = checkpoint_data['results']
                    print("Loaded results from checkpoint")
                else:
                    final_results = process_chunk(list_a, list_b, 1)
                    save_checkpoint(1, final_results, {"chunk_size": len(list_b)})
            else:
                final_results = process_chunk(list_a, list_b, 1)
                save_checkpoint(1, final_results, {"chunk_size": len(list_b)})
        else:
            final_results = process_chunk(list_a, list_b, 1)
            save_checkpoint(1, final_results, {"chunk_size": len(list_b)})
    
    # Write results to tab-delimited file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['List_A_Item', 'Best_Match_from_List_B'])
        writer.writerows(final_results)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    matched_count = sum(1 for result in final_results if result[1] != "none")
    print(f"Summary: {matched_count}/{len(list_a)} items matched")
    
    # Clean up checkpoints after successful completion
    response = input("Processing completed successfully. Clean up checkpoint files? (y/n): ").lower().strip()
    if response == 'y':
        clean_checkpoints()
        print("Checkpoints cleaned up")

def process_chunk(list_a: List[str], list_b_chunk: List[str], chunk_num: int) -> List[List[str]]:
    """
    Process a single chunk of list_b against all items in list_a.
    """
    cached_system_prompt = create_cached_system_prompt(list_b_chunk)
    estimated_tokens = estimate_tokens(cached_system_prompt)
    
    print(f"Chunk {chunk_num}: Creating cached system prompt with ~{estimated_tokens} tokens...")
    
    # Initial request to warm up cache
    last_request_time = wait_for_rate_limit(estimated_tokens, 0)
    test_result = find_best_match_cached("test item", cached_system_prompt)
    print(f"Chunk {chunk_num}: Cache warm-up complete. Test result: {test_result}")
    
    results = []
    
    for i, item_a in enumerate(list_a, 1):
        print(f"Chunk {chunk_num}: Processing item {i}/{len(list_a)}: {item_a[:50]}...")
        
        # Rate limiting for subsequent requests (much smaller token count)
        last_request_time = wait_for_rate_limit(estimate_tokens(item_a) + 50, last_request_time)
        
        match = find_best_match_cached(item_a, cached_system_prompt)
        validated_match = validate_match(match, list_b_chunk)
        
        results.append([item_a, validated_match])
        
        time.sleep(RATE_LIMIT_DELAY)
    
    return results

def choose_best_match_from_candidates(item_a: str, candidates: List[str], client) -> str:
    """
    Use Claude to choose the best match from a list of candidate matches.
    """
    candidates_str = "\n".join([f"{i+1}. {candidate}" for i, candidate in enumerate(candidates)])
    
    system_prompt = """You are a precise text matching assistant. Your task is to choose the best semantic match for a given item from a list of candidate matches.

Instructions:
- Return only the exact text of the best matching candidate from the list provided
- Consider semantic similarity, synonyms, nutrient composition, and conceptual relationships
- Choose the candidate that is most similar in meaning to the target item
- Return only the exact text of the best matching item (without the number prefix)
- Your response should contain ONLY the selected candidate text, nothing else"""
    
    user_prompt = f"""Choose the best match for: "{item_a}"

Candidate matches:
{candidates_str}"""
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=100,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            result = response.content[0].text.strip().strip('"\'') 
            return result
                
        except anthropic.RateLimitError as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Rate limit exceeded when choosing best match for '{item_a}' after {max_retries} attempts: {e}")
                return candidates[0]  # Return first candidate as fallback
            
            wait_time = 60 * retry_count
            print(f"Rate limit hit when choosing best match. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"Error choosing best match for '{item_a}': {e}")
            return candidates[0]  # Return first candidate as fallback
    
    print(f"Warning: Something went wrong. Using first candidate as fallback.")
    return candidates[0]  # Fallback to first candidate


def combine_chunk_results(list_a: List[str], all_chunk_results: List[List[List[str]]], client) -> List[List[str]]:
    """
    Combine results from multiple chunks. When multiple chunks return matches for the same item,
    re-prompt Claude to choose the best match from among the candidates.
    """
    final_results = []
    
    for i, item_a in enumerate(list_a):
        # Collect all non-"none" matches for this item across chunks
        candidate_matches = []
        
        # all_chunk_results = [
        #  [["item1", "match1"], ["item2", "none"], ...],  # chunk 1 results
        #  [["item1", "match2"], ["item2", "match3"], ...], # chunk 2 results  
        #  [["item1", "none"], ["item2", "match4"], ...]    # chunk 3 results
        # ]
        for chunk_results in all_chunk_results:
            if i < len(chunk_results) and chunk_results[i][1] != "none":
                candidate_matches.append(chunk_results[i][1])
        
        if len(candidate_matches) == 0:
            # No matches found in any chunk
            best_match = "none"
        elif len(candidate_matches) == 1:
            # Only one match found, use it
            best_match = candidate_matches[0]
        else:
            # Multiple matches found, re-prompt Claude to choose the best one
            print(f"Item '{item_a[:50]}...' has {len(candidate_matches)} candidate matches. Re-prompting for best choice...")
            best_match = choose_best_match_from_candidates(item_a, candidate_matches, client)
        
        final_results.append([item_a, best_match])
    
    return final_results


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
    list_a, list_b = load_lists_from_files('input_desc_list_noquotes.txt', 'target_desc_list_noquotes.txt')
    
    print(f"Loaded {len(list_a)} items to match against {len(list_b)} target items")
    
    # Perform matching with Claude Sonnet 4 and prompt caching
    match_lists_with_caching(list_a, list_b, OUTPUT_FILE)
    
    print(f"\nDone! Check {OUTPUT_FILE} for results.")
