import argparse
import sys
import subprocess
import json
import os
from pathlib import Path

from prompts import get_all_prompts
from exp_datasets import get_dataset, get_gts
from models import LLM

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

results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

def get_results_filename(dataset, lang, cot):
    cot_str = '_cot' if cot else ''
    return f'results/{dataset}_{lang}{cot_str}_results.json'

def get_prompts_gts_filename(dataset, lang, cot):
    cot_str = '_cot' if cot else ''
    return f'results/{dataset}_{lang}{cot_str}_prompts_gts.json'

def load_existing_results(dataset, lang, cot):
    filename = get_results_filename(dataset, lang, cot)
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {filename}, starting with fresh results")
            return {}
    return {}

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

dataset_data = get_dataset(dataset)
all_prompts = get_all_prompts(dataset, data=dataset_data, lang=lang, cot=cot)
gts = get_gts(dataset)

prompts_gts = load_or_save_prompts_gts(dataset, lang, cot, all_prompts, gts)

max_new_tokens = 2048

llm_models = [
    "DEEPSEEK-R1",
    "DEEPSEEK-R1-DISTILL-QWEN-7B-GGUF",
]

gen_params = {"max_new_tokens": max_new_tokens}

subprocess.run(["gpustat"])

def print_output(reasoning_steps, answer, gt):
    print("---")
    print(reasoning_steps)
    print("---")
    print(answer)
    print("---")
    print(gt)
    print("---")
    sys.stdout.flush()

results = load_existing_results(dataset, lang, cot)

for llm_name in llm_models:
    print(f"\nProcessing LLM: {llm_name}\n")
    
    llm = LLM(llm_name, gen_params=gen_params)
    
    if dataset == "emobench":
        print("Running EmoBench experiment...")
        
        if llm_name not in results:
            results[llm_name] = {
                "EA": {
                    "reasoning_steps": [],
                    "answers": []
                },
                "EU": {
                    "Emotion": {"reasoning_steps": [], "answers": []},
                    "Cause": {"reasoning_steps": [], "answers": []}
                }
            }
        
        print("Emobench has two categories: EA and EU")
        
        print("Starting with EA: ")
        llm_results = results[llm_name]["EA"]
        completed_samples = len(llm_results.get("answers", []))
        total_samples = len(all_prompts["EA"])
        
        if completed_samples < total_samples:
            print(f"Continuing EA experiments for {llm_name}: {completed_samples}/{total_samples} completed")
            
            for i, (prompt, gt) in enumerate(zip(prompts_gts["EA"]["prompts"][completed_samples:], prompts_gts["EA"]["gts"][completed_samples:])):
                print(f"Processing sample {completed_samples + i + 1}/{total_samples}")
                
                print(prompt[1]["content"])
                print("---")
                answer = llm.generate(prompt)
                reasoning_steps, answer = llm.parse_think_output(answer)
                print_output(reasoning_steps, answer, gt)
                print("---")
                sys.stdout.flush()
                
                llm_results["reasoning_steps"].append(reasoning_steps)
                llm_results["answers"].append(answer)
                save_results(results, dataset, lang, cot)
        else:
            print(f"EA experiments already completed for {llm_name}")
        
        print("\nFinished EA!\n")
        print("\nEU has two subcategories: Emotion and Cause")
        
        print("Starting with Emotion: ")
        llm_results = results[llm_name]["EU"]["Emotion"]
        completed_samples = len(llm_results.get("answers", []))
        total_samples = len(all_prompts["EU"]["Emotion"])
        
        if completed_samples < total_samples:
            print(f"Continuing EU-Emotion experiments for {llm_name}: {completed_samples}/{total_samples} completed")
            
            for i, (prompt, gt) in enumerate(zip(prompts_gts["EU"]["Emotion"]["prompts"][completed_samples:], prompts_gts["EU"]["Emotion"]["gts"][completed_samples:])):
                print(f"Processing sample {completed_samples + i + 1}/{total_samples}")
                
                print(prompt[1]["content"])
                print("---")
                answer = llm.generate(prompt)
                reasoning_steps, answer = llm.parse_think_output(answer)
                print_output(reasoning_steps, answer, gt)
                print("---")
                sys.stdout.flush()
                
                llm_results["reasoning_steps"].append(reasoning_steps)
                llm_results["answers"].append(answer)
                save_results(results, dataset, lang, cot)
        else:
            print(f"EU-Emotion experiments already completed for {llm_name}")
            
        print("\nFinished Emotion!\n")
        
        print("Starting with Cause: ")
        llm_results = results[llm_name]["EU"]["Cause"]
        completed_samples = len(llm_results.get("answers", []))
        total_samples = len(prompts_gts["EU"]["Cause"]["prompts"])
        
        if completed_samples < total_samples:
            print(f"Continuing EU-Cause experiments for {llm_name}: {completed_samples}/{total_samples} completed")
            
            for i, (prompt, gt) in enumerate(zip(prompts_gts["EU"]["Cause"]["prompts"][completed_samples:], prompts_gts["EU"]["Cause"]["gts"][completed_samples:])):
                print(f"Processing sample {completed_samples + i + 1}/{total_samples}")
                
                print(prompt[1]["content"])
                print("---")
                answer = llm.generate(prompt)
                reasoning_steps, answer = llm.parse_think_output(answer)
                print_output(reasoning_steps, answer, gt)
                print("---")
                sys.stdout.flush()
                
                llm_results["reasoning_steps"].append(reasoning_steps)
                llm_results["answers"].append(answer)
                save_results(results, dataset, lang, cot)
        else:
            print(f"EU-Cause experiments already completed for {llm_name}")
            
        print("\nFinished Cause!\n")
        
    elif dataset == "tombench":
        print("Running TomBench experiment...")
        print(f"TomBench has {len(prompts_gts)} categories: {list(prompts_gts.keys())}")
        
        if llm_name not in results:
            results[llm_name] = {}
            for category in prompts_gts.keys():
                results[llm_name][category] = {
                    "reasoning_steps": [],
                    "answers": []
                }
    
        for category in prompts_gts.keys():
            if category not in results[llm_name]:
                results[llm_name][category] = {
                    "reasoning_steps": [],
                    "answers": []
                }
                
            llm_results = results[llm_name][category]
            completed_samples = len(llm_results.get("answers", []))
            total_samples = len(prompts_gts[category]["prompts"])
            
            print(f"\nStarting with {category}")
            
            if completed_samples < total_samples:
                print(f"Continuing {category} experiments for {llm_name}: {completed_samples}/{total_samples} completed")
                
                for i, (prompt, gt) in enumerate(zip(prompts_gts[category]["prompts"][completed_samples:], prompts_gts[category]["gts"][completed_samples:])):
                    print(f"Processing sample {completed_samples + i + 1}/{total_samples}")
                    
                    print(prompt[1]["content"])
                    print("---")
                    answer = llm.generate(prompt)
                    reasoning_steps, answer = llm.parse_think_output(answer)
                    print_output(reasoning_steps, answer, gt)
                    print("---")
                    sys.stdout.flush()
                    
                    llm_results["reasoning_steps"].append(reasoning_steps)
                    llm_results["answers"].append(answer)
                    save_results(results, dataset, lang, cot)
            else:
                print(f"{category} experiments already completed for {llm_name}")
else:
    raise ValueError("Invalid dataset: {}".format(dataset))