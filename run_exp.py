import argparse
import sys
from pathlib import Path
from collections import defaultdict

from prompts import get_all_prompts
from exp_datasets import get_gts, get_dataset, get_emo_eu_cats
from models import LLM
from utils import load_existing_results, save_results, save_prompts

parser = argparse.ArgumentParser(description='Run experiment with language model')
parser.add_argument('-l', '--lang', type=str, choices=['en', 'zh'], default='en',
                    help='Language for the experiment (en or zh)')
parser.add_argument('--cot', action='store_true', default=False,
                    help='Whether to use chain-of-thought prompting')
parser.add_argument('-d', '--dataset', type=str, choices=['emobench', 'tombench'], default='emobench',
                    help='Dataset to use (emobench or tombench)')

args = parser.parse_args()
lang = args.lang
cot = args.cot
dataset = args.dataset


llm_models = ["DEEPSEEK-R1", "DEEPSEEK-R1-DISTILL-QWEN-7B", "QWEN-QwQ-32B-GGUF"]

# Get the dataset data
dataset_data = get_dataset(dataset)

# Load ground truths
gts = get_gts(dataset)

saved_prompts = get_all_prompts(dataset, data=dataset_data, lang=lang, cot=cot)
save_prompts(dataset, lang, cot, saved_prompts)

gen_params = {"temperature": 0.6, "max_tokens": 4096}

results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# Load existing results
results = load_existing_results(dataset, lang, cot)

# Main experiment loop
for llm_name in llm_models:
    print(f"\nProcessing LLM: {llm_name}\n")
    
    # Set add_think for DISTILL models
    add_think = True if "DISTILL" in llm_name else False
    
    # Get prompts with appropriate add_think setting for this model
    all_prompts = get_all_prompts(dataset, data=dataset_data, lang=lang, cot=cot, add_think=add_think)
    
    # Initialize model results if not already present
    if llm_name not in results:
        if dataset == "emobench":
            results[llm_name] = {
                "EA": [],
                "EU": {
                    "Emotion": [],
                    "Cause": []
                }
            }
        else:  # tombench
            results[llm_name] = []
    
    if dataset == "emobench":
        # Process EA samples
        if "EA" in all_prompts:
            ea_prompts = all_prompts["EA"]
            completed_samples = len(results[llm_name]["EA"])
            total_samples = len(ea_prompts)
            
            if completed_samples < total_samples:
                print(f"Continuing EA experiments for {llm_name}: {completed_samples}/{total_samples} completed")
                
                # Get problem and relationship data
                from exp_datasets import get_emo_ea_problems_and_relationships
                ea_problems, ea_relationships = get_emo_ea_problems_and_relationships()
                
                llm = LLM(llm_name, gen_params=gen_params)
                
                for i, (prompt, gt) in enumerate(zip(ea_prompts[completed_samples:], gts["EA"][completed_samples:])):
                    print(f"Processing sample {completed_samples + i + 1}/{total_samples}")
                    
                    # Get problem and relationship for this sample
                    problem = ea_problems[completed_samples + i]
                    relationship = ea_relationships[completed_samples + i]
                    
                    output = llm.generate(prompt)
                    reasoning_steps, answer = llm.parse_think_output(output)
                    sys.stdout.flush()
                    
                    # Create sample with flattened structure
                    sample = {
                        "answer": answer,
                        "reasoning": reasoning_steps,
                        "problem": problem,
                        "relationship": relationship
                    }
                    
                    # Add to results
                    results[llm_name]["EA"].append(sample)
                    save_results(results, dataset, lang, cot)
            else:
                print(f"EA experiments already completed for {llm_name}")
        
        # Process EU-Emotion samples
        if "EU" in all_prompts and "Emotion" in all_prompts["EU"]:
            eu_emotion_prompts = all_prompts["EU"]["Emotion"]
            completed_samples = len(results[llm_name]["EU"]["Emotion"])
            total_samples = len(eu_emotion_prompts)
            
            if completed_samples < total_samples:
                print(f"Continuing EU-Emotion experiments for {llm_name}: {completed_samples}/{total_samples} completed")
                
                # Get category information
                from exp_datasets import get_emo_eu_cats
                eu_categories = get_emo_eu_cats()
                
                llm = LLM(llm_name, gen_params=gen_params)
                
                for i, (prompt, gt) in enumerate(zip(eu_emotion_prompts[completed_samples:], gts["EU"]["Emotion"][completed_samples:])):
                    print(f"Processing sample {completed_samples + i + 1}/{total_samples}")
                    
                    # Get category for this sample (index matches the sample index)
                    category = eu_categories[completed_samples + i]
                    
                    output = llm.generate(prompt)
                    reasoning_steps, answer = llm.parse_think_output(output)
                    sys.stdout.flush()
                    
                    # Create sample with flattened structure
                    sample = {
                        "answer": answer,
                        "reasoning": reasoning_steps,
                        "category": category
                    }
                    
                    # Add to results
                    results[llm_name]["EU"]["Emotion"].append(sample)
                    save_results(results, dataset, lang, cot)
            else:
                print(f"EU-Emotion experiments already completed for {llm_name}")
        
        # Process EU-Cause samples
        if "EU" in all_prompts and "Cause" in all_prompts["EU"]:
            eu_cause_prompts = all_prompts["EU"]["Cause"]
            completed_samples = len(results[llm_name]["EU"]["Cause"])
            total_samples = len(eu_cause_prompts)
            
            if completed_samples < total_samples:
                print(f"Continuing EU-Cause experiments for {llm_name}: {completed_samples}/{total_samples} completed")
                
                # Get category information
                from exp_datasets import get_emo_eu_cats
                eu_categories = get_emo_eu_cats()
                
                llm = LLM(llm_name, gen_params=gen_params)
                
                for i, (prompt, gt) in enumerate(zip(eu_cause_prompts[completed_samples:], gts["EU"]["Cause"][completed_samples:])):
                    print(f"Processing sample {completed_samples + i + 1}/{total_samples}")
                    
                    # Get category for this sample (index matches the sample index)
                    category = eu_categories[completed_samples + i]
                    
                    output = llm.generate(prompt)
                    reasoning_steps, answer = llm.parse_think_output(output)
                    sys.stdout.flush()
                    
                    # Create sample with flattened structure
                    sample = {
                        "answer": answer,
                        "reasoning": reasoning_steps,
                        "category": category
                    }
                    
                    # Add to results
                    results[llm_name]["EU"]["Cause"].append(sample)
                    save_results(results, dataset, lang, cot)
            else:
                print(f"EU-Cause experiments already completed for {llm_name}")
                
        print("\nFinished EmoBench!\n")
        
    elif dataset == "tombench":
        print("Running TomBench experiment...")
        print(f"TomBench has {len(all_prompts)} categories: {list(all_prompts.keys())}")
        
        # Get the tombench dataset for ability information
        tombench_data = get_dataset("tombench")
        
        # Get completed samples count
        completed_samples = len(results[llm_name])
        
        # Calculate total samples across all categories
        total_samples = sum(len(all_prompts[category]) for category in all_prompts.keys())
        
        # Track samples we've already processed
        processed_samples = defaultdict(int)
        
        for category in all_prompts.keys():
            category_prompts = all_prompts[category]
            category_gts = gts[category]
            
            # Count how many samples we've already processed for this category
            category_samples_done = 0
            for sample in results[llm_name]:
                if sample["topic"] == category:
                    category_samples_done += 1
            
            processed_samples[category] = category_samples_done
            
            # Check if we need to process more samples for this category
            if category_samples_done < len(category_prompts):
                print(f"\nContinuing {category} experiments for {llm_name}: {category_samples_done}/{len(category_prompts)} completed")
                
                llm = LLM(llm_name, gen_params=gen_params)
                
                # Process remaining samples for this category
                for i, (prompt, gt) in enumerate(zip(category_prompts[category_samples_done:], category_gts[category_samples_done:])):
                    sample_index = category_samples_done + i
                    
                    print(f"Processing sample {sample_index + 1}/{len(category_prompts)}")
                    
                    output = llm.generate(prompt)
                    reasoning_steps, answer = llm.parse_think_output(output)
                    sys.stdout.flush()
                    
                    # Get ability information from the dataset
                    ability_info = {}
                    category_dataset = tombench_data[category]
                    
                    if sample_index < len(category_dataset) and "能力\nABILITY" in category_dataset[sample_index]:
                        ability = category_dataset[sample_index]["能力\nABILITY"]
                        
                        # Parse ability information
                        if "location false beliefs belief: second-order beliefs" in ability.lower():
                            main_ability = "belief"
                            sub_ability = "second-order beliefs"
                        elif "content false beliefs belief: second-order beliefs" in ability.lower():
                            main_ability = "belief"
                            sub_ability = "second-order beliefs"
                        elif ":" in ability:
                            main_ability, sub_ability = ability.split(":", 1)
                            main_ability = main_ability.strip().lower()
                            sub_ability = sub_ability.strip().lower()
                            
                            if sub_ability == "desires influence on actions" or sub_ability == "desires influence on emotions (beliefs)":
                                sub_ability = "desires influence on actions and emotions"
                        else:
                            main_ability = ability.strip().lower()
                            sub_ability = "general"
                        
                        ability_info = {
                            "main_ability": main_ability,
                            "sub_ability": sub_ability
                        }
                    else:
                        # Default fallback
                        ability_info = {
                            "main_ability": "unknown",
                            "sub_ability": category.lower()
                        }
                    
                    # Create sample with flattened structure
                    sample = {
                        "topic": category,
                        "answer": answer,
                        "reasoning": reasoning_steps,
                        "main_ability": ability_info["main_ability"],
                        "sub_ability": ability_info["sub_ability"]
                    }
                    
                    results[llm_name].append(sample)
                    save_results(results, dataset, lang, cot)
                    
                    processed_samples[category] += 1
            else:
                print(f"{category} experiments already completed for {llm_name}")
        
        print("\nFinished TomBench!\n")
    
    else:
        raise ValueError("Invalid dataset: {}".format(dataset))

print("All experiments completed!")
