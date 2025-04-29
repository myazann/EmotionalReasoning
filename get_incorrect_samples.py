import json
import argparse
import re
import pandas as pd
from pathlib import Path
from exp_datasets import get_gts, get_emo_data, get_tom_data

def get_emobench_prediction_index(pred_raw):
    """Attempt to convert raw prediction (string, letter, number) to a 0-based integer index.
       Returns integer index or None if conversion fails.
       
       For EmoBench:
       - Most predictions are just a number (0, 1, 2, 3) which needs to be converted to an index
       - Some predictions may be a lengthy text ending with a number
       - Rarely, they may be letters (A, B, C, D) instead of numbers
    """
    if pred_raw is None or pd.isna(pred_raw):
        return None
        
    if not isinstance(pred_raw, (str, int)):
        return None

    pred = str(pred_raw).strip()
    
    # If empty after stripping, return None
    if not pred:
        return None
    
    # First try to extract a number from the very end of the string
    # This handles cases where the model gives a verbose answer and then the final digit
    digit_at_end = re.search(r'(\d+)\s*$', pred)
    if digit_at_end:
        return int(digit_at_end.group(1))
    
    # Try to extract any digit in the string
    any_digit = re.search(r'\b(\d+)\b', pred)
    if any_digit:
        return int(any_digit.group(1))
    
    # Check if the whole thing is a digit
    if pred.isdigit():
        return int(pred)
    
    # If it was already an integer
    if isinstance(pred_raw, int):
        return pred_raw
        
    # Try to handle letter answers - first look for single letter answer (rare in EmoBench)
    if re.match(r'^[A-Za-z]$', pred):  
        if 'A' <= pred.upper() <= 'F':
            return ord(pred.upper()) - ord('A')
    
    # Check for a letter at the end (rare in EmoBench)
    letter_at_end = re.search(r'([A-Za-z])\s*$', pred)
    if letter_at_end:
        last_char = letter_at_end.group(1)
        if 'A' <= last_char.upper() <= 'F':
            return ord(last_char.upper()) - ord('A')
            
    # Check for bracketed format just in case
    bracket_match = re.search(r'\[\[([A-Za-z])\]\]', pred)
    if bracket_match:
        letter = bracket_match.group(1)
        if 'A' <= letter.upper() <= 'F':
            return ord(letter.upper()) - ord('A')
            
    # Couldn't parse a valid index
    return None

def get_tombench_prediction_index(pred_raw, available_choices=None):
    """Attempt to convert raw prediction (string, letter, number) to a 0-based integer index.
       Returns integer index or None if conversion fails.
       
       For TomBench:
       - Predictions are usually in the format [[A]], [[B]], etc.
       - Sometimes there might be text before the double brackets
       - NaN values may be present in the data
       - Some questions have fewer than 4 choices
    """
    if pred_raw is None or pd.isna(pred_raw):
        return None
        
    if not isinstance(pred_raw, (str, int)):
        return None

    pred = str(pred_raw).strip()
    
    # If empty after stripping, return None
    if not pred:
        return None
        
    # Convert prediction to letter first, then to index
    predicted_letter = None
    
    # PRIMARY METHOD: Extract letter from [[X]] format (standard TomBench format)
    bracket_match = re.search(r'\[\[([A-Za-z])\]\]', pred)
    if bracket_match:
        predicted_letter = bracket_match.group(1).upper()
    
    # Check for multiple [[X]] patterns and use the last one if present
    if not predicted_letter:
        bracket_matches = re.findall(r'\[\[([A-Za-z])\]\]', pred)
        if bracket_matches and len(bracket_matches) > 0:
            predicted_letter = bracket_matches[-1].upper()  # Use the last one if multiple matches
            
    # Try looking for any single letter (A-F) in the string
    if not predicted_letter:
        letter_match = re.search(r'\b([A-F])\b', pred, re.IGNORECASE)
        if letter_match:
            predicted_letter = letter_match.group(1).upper()
    
    # Check for a single letter answer
    if not predicted_letter and re.match(r'^[A-Za-z]$', pred):
        predicted_letter = pred.upper()
    
    # Check for a letter at the end
    if not predicted_letter:
        letter_at_end = re.search(r'([A-Za-z])\s*$', pred)
        if letter_at_end:
            predicted_letter = letter_at_end.group(1).upper()
            
    # Convert letter to index if we found one
    if predicted_letter and 'A' <= predicted_letter <= 'F':
        # Convert letter to index (A=0, B=1, etc.)
        letter_idx = ord(predicted_letter) - ord('A')
        
        # If we know the available choices, check if the index is valid
        if available_choices is not None:
            if letter_idx < len(available_choices):
                return letter_idx
            else:
                # If prediction is out of range for available choices, return None
                return None
        else:
            # No choice validation, just return the index
            return letter_idx
            
    # Try numeric formats even though they're rare in TomBench
    if pred.isdigit():
        numeric_idx = int(pred)
        # Validate against available choices if provided
        if available_choices is not None and numeric_idx >= len(available_choices):
            return None
        return numeric_idx
        
    digit_match = re.search(r'\b(\d)\b', pred)
    if digit_match:
        numeric_idx = int(digit_match.group(1))
        # Validate against available choices if provided
        if available_choices is not None and numeric_idx >= len(available_choices):
            return None
        return numeric_idx
    
    # If it was already an integer
    if isinstance(pred_raw, int):
        # Validate against available choices if provided
        if available_choices is not None and pred_raw >= len(available_choices):
            return None
        return pred_raw

    # Failed to convert
    return None

def extract_letter_from_answer(answer):
    """Extract the letter answer (A, B, C, D) from the ground truth answer."""
    if isinstance(answer, str) and len(answer) == 1 and 'A' <= answer.upper() <= 'D':
        return answer.upper()
    
    # Try to find [[X]] pattern
    bracket_match = re.search(r'\[\[([A-Za-z])\]\]', answer)
    if bracket_match:
        return bracket_match.group(1).upper()
    
    return answer

def process_tombench_scenarios(model_gens, gts_data, tombench_data, lang="en"):
    """Process TomBench results for a specific model."""
    all_samples = []
    
    # Group model generations by scenario name and keep track of their indices within each scenario
    scenario_questions = {}
    for idx, sample in enumerate(model_gens):
        scenario_name = sample.get("topic")
        if scenario_name not in scenario_questions:
            scenario_questions[scenario_name] = []
        scenario_questions[scenario_name].append((idx, sample))
    
    # Process each scenario
    for scenario_name, scenario_gts in gts_data.items():
        # Get the original data for this scenario
        scenario_data = tombench_data.get(scenario_name, [])
        
        # Skip if we don't have this scenario in our generations
        if scenario_name not in scenario_questions:
            continue
        
        # Get model's responses for this scenario
        scenario_samples = scenario_questions[scenario_name]
        
        # Make sure the number of model generations matches the number of questions in the scenario
        if len(scenario_samples) != len(scenario_gts):
            print(f"Warning: Mismatch in number of questions for {scenario_name}. "
                  f"Found {len(scenario_samples)} model predictions but {len(scenario_gts)} ground truths.")
        
        # Process each question with its corresponding model generation
        for idx, gt_answer in enumerate(scenario_gts):
            if idx >= len(scenario_data) or idx >= len(scenario_samples):
                continue  # Skip if we don't have this question's data or model generation
            
            # Get the model's generation for this specific question
            _, sample = scenario_samples[idx]
            
            # Get the sample data
            sample_data = scenario_data[idx]
            
            # Get options/choices
            choices = []
            for option_letter in ["A", "B", "C", "D"]:
                en_option_key = f"OPTION-{option_letter}"
                zh_option_key = f"选项{option_letter}"
                
                option_key = en_option_key if lang == "en" else zh_option_key
                if option_key in sample_data and pd.notna(sample_data[option_key]):
                    choices.append(sample_data[option_key])
            
            # If no choices were found, try again with a different approach
            if not choices:
                print(f"Warning: No choices found for {scenario_name}, question {idx}. Trying alternate method.")
                # Some questions might be in a different format
                for key, value in sample_data.items():
                    if (key.startswith("OPTION-") or key.startswith("选项")) and pd.notna(value):
                        choices.append(value)
            
            # Extract prediction and convert to index, passing available choices
            raw_pred = sample.get("answer")
            pred_idx = get_tombench_prediction_index(raw_pred, available_choices=choices)
            
            # Convert ground truth to index, passing available choices
            gt_letter = extract_letter_from_answer(gt_answer)
            true_idx = get_tombench_prediction_index(gt_letter, available_choices=choices)
            
            # If we have exactly two choices and predictions or ground truths contain "A" or "B",
            # make sure indices are properly restricted to 0-1
            if len(choices) == 2:
                # For ground truth
                if isinstance(gt_letter, str) and gt_letter.upper() in ['A', 'B']:
                    true_idx = ord(gt_letter.upper()) - ord('A')
                elif isinstance(true_idx, int) and true_idx > 1:  # If somehow index is > 1 for 2 choices
                    true_idx = None  # Mark as invalid
                    
                # For prediction
                if isinstance(raw_pred, str) and re.search(r'\b[AB]\b', raw_pred.upper()):
                    letter_match = re.search(r'\b([AB])\b', raw_pred.upper())
                    if letter_match:
                        pred_idx = ord(letter_match.group(1)) - ord('A')
                elif isinstance(pred_idx, int) and pred_idx > 1:  # If somehow index is > 1 for 2 choices
                    pred_idx = None  # Mark as invalid
            
            # Set prediction and ground truth labels
            true_label = "Unknown"
            pred_label = "Invalid/Unknown Prediction"
            
            # Handle cases where questions might have fewer choices
            if choices and true_idx is not None:
                if 0 <= true_idx < len(choices):
                    true_label = choices[true_idx]
                else:
                    # If true_idx is out of range, log the issue
                    true_label = f"Index {true_idx} out of range (only {len(choices)} choices available)"
                    
            if choices and pred_idx is not None:
                if 0 <= pred_idx < len(choices):
                    pred_label = choices[pred_idx]
                else:
                    # If pred_idx is out of range, log the issue
                    pred_label = f"Index {pred_idx} out of range (only {len(choices)} choices available)"
            
            # Get the story and question text based on language
            story_key = "STORY" if lang == "en" else "故事"
            question_key = "QUESTION" if lang == "en" else "问题"
            
            story = sample_data.get(story_key, "Story text not found")
            question = sample_data.get(question_key, "Question text not found")
            
            # Get ability information
            ability_key = "能力\nABILITY" 
            ability = sample_data.get(ability_key, "Unknown")
            
            # Collect additional abilities from the model generation
            main_ability = sample.get("main_ability", "")
            sub_ability = sample.get("sub_ability", "")
            
            # Determine if the prediction is correct
            is_correct = True
            if pred_idx is None or pred_idx != true_idx:
                is_correct = False
                
            all_samples.append({
                "scenario": scenario_name,
                "ability": ability,
                "main_ability": main_ability,
                "sub_ability": sub_ability,
                "story": story,
                "question": question,
                "choices": choices,
                "ground_truth_idx": true_idx,
                "ground_truth_letter": gt_letter,
                "ground_truth_label": true_label,
                "prediction_idx": pred_idx,
                "prediction_label": pred_label,
                "raw_prediction": raw_pred,
                "reasoning": sample.get("reasoning", ""),
                "is_correct": is_correct
            })
    
    return all_samples

def process_ea_task(model_gens, ea_gts, ea_data, lang="en"):
    """Process EA task results for a specific model."""
    all_samples = []
    ea_samples = model_gens.get("EA", [])
    
    for idx, sample in enumerate(ea_samples):
        raw_pred = sample.get("answer")
        pred_idx = get_emobench_prediction_index(raw_pred) # Returns int index or None
        true_idx = ea_gts[idx] # This is an int
        
        choices = ea_data[idx]["Choices"][lang]
        true_label = ea_data[idx]["Label_str"][lang]
        pred_label = "Invalid/Unknown Prediction"

        # Get prediction label if index is valid
        if pred_idx is not None and 0 <= pred_idx < len(choices):
            pred_label = choices[pred_idx]
        
        # Check if the prediction is incorrect by comparing both indices and labels
        # Only mark as incorrect if both the index doesn't match AND the label doesn't match (or prediction is invalid)
        is_correct = True
        if pred_idx is None:
            # Invalid prediction
            is_correct = False
        elif pred_idx != true_idx:
            # Indices don't match, but check if the labels match
            if pred_label != true_label:
                is_correct = False
                
        all_samples.append({
            "task": "EA",
            "problem": sample.get("problem"),
            # Use the 'Scenario' key for the full question text
            "scenario": ea_data[idx].get("Scenario", {}).get(lang, "Scenario text not found"), 
            "choices": choices,
            "ground_truth_idx": true_idx,
            "ground_truth_label": true_label,
            "prediction_idx": pred_idx, # Could be None
            "prediction_label": pred_label,
            "raw_prediction": raw_pred,
            "reasoning": sample.get("reasoning"),
            "is_correct": is_correct
        })
    
    return all_samples

def process_eu_emotion_task(model_gens, emo_gts, eu_data, lang="en"):
    """Process EU-Emotion task results for a specific model."""
    all_samples = []
    emo_samples = model_gens.get("EU", {}).get("Emotion", [])
    
    for idx, sample in enumerate(emo_samples):
        raw_pred = sample.get("answer")
        pred_idx = get_emobench_prediction_index(raw_pred)
        true_idx = emo_gts[idx]
        
        choices = eu_data[idx]["Emotion"]["Choices"][lang]
        true_label = eu_data[idx]["Emotion"]["Label"][lang]
        pred_label = "Invalid/Unknown Prediction"
        
        if pred_idx is not None and 0 <= pred_idx < len(choices):
            pred_label = choices[pred_idx]
        
        # Check if the prediction is incorrect by comparing both indices and labels
        # Only mark as incorrect if both the index doesn't match AND the label doesn't match (or prediction is invalid)
        is_correct = True
        if pred_idx is None:
            # Invalid prediction
            is_correct = False
        elif pred_idx != true_idx:
            # Indices don't match, but check if the labels match
            if pred_label != true_label:
                is_correct = False
                
        all_samples.append({
            "task": "EU-Emotion",
            "category": sample.get("category"),
            "scenario": eu_data[idx].get("Scenario", {}).get(lang, "Scenario text not found"),
            "ground_truth_idx": true_idx,
            "ground_truth_label": true_label,
            "prediction_idx": pred_idx,
            "prediction_label": pred_label,
            "raw_prediction": raw_pred,
            "choices": choices,
            "reasoning": sample.get("reasoning"),
            "is_correct": is_correct
        })
    
    return all_samples

def process_eu_cause_task(model_gens, cause_gts, eu_data, lang="en"):
    """Process EU-Cause task results for a specific model."""
    all_samples = []
    cause_samples = model_gens.get("EU", {}).get("Cause", [])
    
    for idx, sample in enumerate(cause_samples):
        raw_pred = sample.get("answer")
        pred_idx = get_emobench_prediction_index(raw_pred)
        true_idx = cause_gts[idx]
        
        choices = eu_data[idx]["Cause"]["Choices"][lang]
        true_label = eu_data[idx]["Cause"]["Label"][lang]
        pred_label = "Invalid/Unknown Prediction"

        if pred_idx is not None and 0 <= pred_idx < len(choices):
             pred_label = choices[pred_idx]

        # Check if the prediction is incorrect by comparing both indices and labels
        # Only mark as incorrect if both the index doesn't match AND the label doesn't match (or prediction is invalid)
        is_correct = True
        if pred_idx is None:
            # Invalid prediction
            is_correct = False
        elif pred_idx != true_idx:
            # Indices don't match, but check if the labels match
            if pred_label != true_label:
                is_correct = False
                
        all_samples.append({
            "task": "EU-Cause",
            "category": sample.get("category"),
            "scenario": eu_data[idx].get("Scenario", {}).get(lang, "Scenario text not found"), 
            "ground_truth_idx": true_idx,
            "ground_truth_label": true_label,
            "prediction_idx": pred_idx,
            "prediction_label": pred_label,
            "raw_prediction": raw_pred,
            "choices": choices,
            "reasoning": sample.get("reasoning"),
            "is_correct": is_correct
        })
    
    return all_samples

def process_emobench(lang, results_dir):
    """Process EmoBench results for a specific language."""
    gens_file = results_dir / f"emobench_{lang}_generations.json"
    if not gens_file.exists():
        print(f"Generations file not found: {gens_file}")
        return

    # Load model generations
    with gens_file.open() as f:
        gens = json.load(f)

    # Load ground truths and original dataset
    gts = get_gts("emobench")
    orig_data = get_emo_data()
    
    # Dictionary to store all examples (both correct and incorrect) for each model
    all_samples_dict = {}
    incorrect_count = {}
    
    # Process each model's data
    for model_key in gens:
        model_gens = gens[model_key]
        model_samples = []
        incorrect_count[model_key] = 0
        
        # Process EA task
        ea_samples = process_ea_task(model_gens, gts["EA"], orig_data["EA"], lang)
        for sample in ea_samples:
            if not sample.get("is_correct", False):
                incorrect_count[model_key] += 1
        model_samples.extend(ea_samples)
        
        # Process EU-Emotion task
        emo_samples = process_eu_emotion_task(model_gens, gts["EU"]["Emotion"], orig_data["EU"], lang)
        for sample in emo_samples:
            if not sample.get("is_correct", False):
                incorrect_count[model_key] += 1
        model_samples.extend(emo_samples)
        
        # Process EU-Cause task
        cause_samples = process_eu_cause_task(model_gens, gts["EU"]["Cause"], orig_data["EU"], lang)
        for sample in cause_samples:
            if not sample.get("is_correct", False):
                incorrect_count[model_key] += 1
        model_samples.extend(cause_samples)
        
        # Add this model's results to the dictionary
        all_samples_dict[model_key] = model_samples
        
        # Also print a summary for this model
        print(f"Model {model_key} has {incorrect_count[model_key]} incorrect samples out of {len(model_samples)} total in {lang} language")

    # Output file with all model results
    out_file = results_dir / f"emobench_{lang}_samples.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(all_samples_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved all samples for all models to {out_file}")

def process_tombench(lang, results_dir):
    """Process TomBench results for a specific language."""
    gens_file = results_dir / f"tombench_{lang}_generations.json"
    if not gens_file.exists():
        print(f"Generations file not found: {gens_file}")
        return

    # Load model generations
    with gens_file.open() as f:
        gens = json.load(f)

    # Load ground truths and original dataset
    gts = get_gts("tombench")
    tombench_data = get_tom_data()
    
    # Dictionary to store all examples (both correct and incorrect) for each model
    all_samples_dict = {}
    incorrect_count = {}
    
    # Process each model's data
    for model_key in gens:
        model_gens = gens[model_key]
        
        # Process TomBench data
        model_samples = process_tombench_scenarios(model_gens, gts, tombench_data, lang)
        
        # Count incorrect samples
        incorrect_count[model_key] = sum(1 for sample in model_samples if not sample.get("is_correct", False))
        
        # Add this model's results to the dictionary
        all_samples_dict[model_key] = model_samples
        
        # Also print a summary for this model
        print(f"Model {model_key} has {incorrect_count[model_key]} incorrect samples out of {len(model_samples)} total in {lang} language")

    # Output file with all model results
    out_file = results_dir / f"tombench_{lang}_samples.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(all_samples_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved all samples for all models to {out_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract incorrect predictions from EmoBench or TomBench results')
    parser.add_argument('-d', '--dataset', type=str, choices=['emobench', 'tombench'], required=True,
                        help='Dataset to process (emobench or tombench)')
    parser.add_argument('-l', '--lang', type=str, choices=['en', 'zh'], default='both',
                        help='Language to process (en, zh, or both)')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    
    # Process the specified dataset(s) and language(s)
    if args.lang == 'both' or args.lang == 'en':
        if args.dataset == 'emobench':
            process_emobench('en', results_dir)
        elif args.dataset == 'tombench':
            process_tombench('en', results_dir)
    
    if args.lang == 'both' or args.lang == 'zh':
        if args.dataset == 'emobench':
            process_emobench('zh', results_dir)
        elif args.dataset == 'tombench':
            process_tombench('zh', results_dir)

if __name__ == "__main__":
    main()
