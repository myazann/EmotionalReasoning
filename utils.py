import os
import json

def get_results_filename(dataset, lang, cot):
    cot_str = '_cot' if cot else ''
    return f'results/{dataset}_{lang}{cot_str}_generations.json'

def get_prompts_gts_filename(dataset, lang, cot):
    cot_str = '_cot' if cot else ''
    return f'results/{dataset}_{lang}{cot_str}_prompts_gts.json'

def get_saved_prompts_filename(dataset, lang, cot):
    cot_str = '_cot' if cot else ''
    return f'results/{dataset}_{lang}{cot_str}_prompts.json'

def load_existing_results(dataset, lang, cot):
    # Try with the new '_generations' format first (from get_results_filename)
    filename = get_results_filename(dataset, lang, cot)
    
    # If the file with the new format doesn't exist, try the old format
    if not os.path.exists(filename):
        # Construct the old filename format
        cot_str = '_cot' if cot else ''
        old_filename = f'results/{dataset}_{lang}{cot_str}_generations.json'
        
        if os.path.exists(old_filename):
            print(f"Using legacy results file {old_filename}")
            filename = old_filename
        else:
            print(f"No results file found. Tried {filename} and {old_filename}")
            return {}
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error reading {filename}, starting with fresh results")
        return {}

def save_prompts(dataset, lang, cot, all_prompts):
    filename = get_saved_prompts_filename(dataset, lang, cot)
    if os.path.exists(filename):
        print(f"Prompts file {filename} already exists, not overwriting")
        return
        
    print(f"Saving prompts to {filename}")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_prompts, f, ensure_ascii=False, indent=2)

def load_or_save_prompts_gts(dataset, lang, cot, all_prompts, gts):
    filename = get_prompts_gts_filename(dataset, lang, cot)
    if os.path.exists(filename):
        print(f"Loading prompts and ground truths from {filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"Creating new prompts and ground truths file {filename}")
        prompts_gts = {}
        
        if dataset == "emobench":
            prompts_gts = {
                "EA": {"prompts": all_prompts["EA"], "gts": gts["EA"]},
                "EU": {
                    "Emotion": {"prompts": all_prompts["EU"]["Emotion"], "gts": gts["EU"]["Emotion"]},
                    "Cause": {"prompts": all_prompts["EU"]["Cause"], "gts": gts["EU"]["Cause"]}
                }
            }
        elif dataset == "tombench":
            prompts_gts = {}
            for category in all_prompts.keys():
                prompts_gts[category] = {
                    "prompts": all_prompts[category],
                    "gts": gts[category]
                }
                
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(prompts_gts, f, ensure_ascii=False, indent=2)
            
        return prompts_gts

def save_results(results, dataset, lang, cot):
    filename = get_results_filename(dataset, lang, cot)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)