#!/usr/bin/env python3
import json
import re
from pathlib import Path
from exp_datasets import get_gts, get_emo_data

def get_prediction_index(pred_raw):
    """Attempt to convert raw prediction (string, letter, number) to a 0-based integer index.
       Returns integer index or None if conversion fails.
    """
    if not isinstance(pred_raw, (str, int)):
        return None

    pred = str(pred_raw).strip()

    # Try converting letter (A=0, B=1, etc.)
    if re.match(r'^[A-Za-z]$', pred):
        # Handle only A-D/a-d for typical multiple choice, adjust if needed
        # Also handle up to F for EU Emotion/Cause potentially having 6 options
        if 'A' <= pred.upper() <= 'F': 
            return ord(pred.upper()) - ord('A')
        else:
            return None # Invalid letter

    # Try converting numeric string
    if pred.isdigit():
        return int(pred)
    
    # If it was already an integer
    if isinstance(pred_raw, int):
        return pred_raw

    return None # Failed to convert

def main():
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    gens_file = results_dir / "emobench_en_generations.json"
    if not gens_file.exists():
        print(f"Generations file not found: {gens_file}")
        return

    # Load model generations
    with gens_file.open() as f:
        gens = json.load(f)

    # Load ground truths and original dataset
    gts = get_gts("emobench")
    orig_data = get_emo_data()
    
    lang = "en"  # We're only interested in English results

    model_key = "DEEPSEEK-R1"
    if model_key not in gens:
        print(f"Model '{model_key}' not found in generations.")
        return

    model_gens = gens[model_key]
    incorrect = []

    # --- Process EA task ---
    ea_samples = model_gens.get("EA", [])
    ea_gts = gts["EA"]
    ea_data = orig_data["EA"]
    
    for idx, sample in enumerate(ea_samples):
        raw_pred = sample.get("answer")
        pred_idx = get_prediction_index(raw_pred) # Returns int index or None
        true_idx = ea_gts[idx] # This is an int
        
        choices = ea_data[idx]["Choices"][lang]
        true_label = ea_data[idx]["Label_str"][lang]
        pred_label = "Invalid/Unknown Prediction"

        # Get prediction label if index is valid
        if pred_idx is not None and 0 <= pred_idx < len(choices):
            pred_label = choices[pred_idx]
        
        # Compare integer indices directly
        # Mark as incorrect if prediction is invalid (None) or doesn't match GT
        if pred_idx is None or pred_idx != true_idx:
            incorrect.append({
                "task": "EA",
                "problem": sample.get("problem"),
                # Use the 'Scenario' key for the full question text
                "scenario": ea_data[idx].get("Scenario", {}).get(lang, "Scenario text not found"), 
                "choices": choices,
                "ground_truth_idx": true_idx,
                "ground_truth_label": true_label,
                "prediction_idx": pred_idx, # Could be None
                "prediction_label": pred_label,
                "reasoning": sample.get("reasoning"),
            })

    # --- Process EU-Emotion task ---
    emo_samples = model_gens.get("EU", {}).get("Emotion", [])
    emo_gts = gts["EU"]["Emotion"]
    eu_data = orig_data["EU"]
    
    for idx, sample in enumerate(emo_samples):
        raw_pred = sample.get("answer")
        pred_idx = get_prediction_index(raw_pred)
        true_idx = emo_gts[idx]
        
        choices = eu_data[idx]["Emotion"]["Choices"][lang]
        true_label = eu_data[idx]["Emotion"]["Label"][lang]
        pred_label = "Invalid/Unknown Prediction"
        
        if pred_idx is not None and 0 <= pred_idx < len(choices):
            pred_label = choices[pred_idx]
        
        if pred_idx is None or pred_idx != true_idx:
            incorrect.append({
                "task": "EU-Emotion",
                "category": sample.get("category"),
                "scenario": eu_data[idx].get("Scenario", {}).get(lang, "Scenario text not found"),
                "ground_truth_idx": true_idx,
                "ground_truth_label": true_label,
                "prediction_idx": pred_idx,
                "prediction_label": pred_label,
                "choices": choices,
                "reasoning": sample.get("reasoning"),
            })

    # --- Process EU-Cause task ---
    cause_samples = model_gens.get("EU", {}).get("Cause", [])
    cause_gts = gts["EU"]["Cause"]
    
    for idx, sample in enumerate(cause_samples):
        raw_pred = sample.get("answer")
        pred_idx = get_prediction_index(raw_pred)
        true_idx = cause_gts[idx]
        
        choices = eu_data[idx]["Cause"]["Choices"][lang]
        true_label = eu_data[idx]["Cause"]["Label"][lang]
        pred_label = "Invalid/Unknown Prediction"

        if pred_idx is not None and 0 <= pred_idx < len(choices):
             pred_label = choices[pred_idx]

        if pred_idx is None or pred_idx != true_idx:
            incorrect.append({
                "task": "EU-Cause",
                "category": sample.get("category"),
                "scenario": eu_data[idx].get("Scenario", {}).get(lang, "Scenario text not found"), 
                "ground_truth_idx": true_idx,
                "ground_truth_label": true_label,
                "prediction_idx": pred_idx,
                "prediction_label": pred_label,
                "choices": choices,
                "reasoning": sample.get("reasoning"),
            })

    # Output file
    out_file = results_dir / "emobench_en_deepseek_r1_incorrect.json"
    with out_file.open("w") as f:
        json.dump(incorrect, f, indent=2)

    print(f"Saved {len(incorrect)} incorrect samples to {out_file}")

if __name__ == "__main__":
    main()
