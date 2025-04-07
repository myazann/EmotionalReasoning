import argparse
import sys
import subprocess
from pathlib import Path

from prompts import get_all_prompts
from exp_datasets import get_dataset, get_gts
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

results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

gen_params = {"max_new_tokens": 4096, "temperature": 0.6}
llm_models = [
    "DEEPSEEK-R1-DISTILL-QWEN-7B",
    "QWEN-QwQ-32B-GGUF",
    "DEEPSEEK-R1-DISTILL-LLAMA-8B", 
    "DEEPSEEK-R1",
]

dataset_data = get_dataset(dataset)
gts = get_gts(dataset)
results = load_existing_results(dataset, lang, cot)

for llm_name in llm_models:
    print(f"\nProcessing LLM: {llm_name}\n")
    
    llm = LLM(llm_name, gen_params=gen_params)
    add_think = True if "DISTILL" in llm_name else False
    
    saved_prompts = get_all_prompts(dataset, data=dataset_data, lang=lang, cot=cot)
    save_prompts(dataset, lang, cot, saved_prompts)

    all_prompts = get_all_prompts(dataset, data=dataset_data, lang=lang, cot=cot, add_think=add_think)
        
    if dataset == "emobench":
        print("Running EmoBench experiment...")
        
        if llm_name not in results:
            results[llm_name] = {
                "EA": {
                    "reasoning_steps": [],
                    "answers": [],
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
            
            for i, (prompt, gt) in enumerate(zip(all_prompts["EA"][completed_samples:], gts["EA"][completed_samples:])):
                print(f"Processing sample {completed_samples + i + 1}/{total_samples}")
                output = llm.generate(prompt)

                reasoning_steps, answer = llm.parse_think_output(output)
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
            
            for i, (prompt, gt) in enumerate(zip(all_prompts["EU"]["Emotion"][completed_samples:], gts["EU"]["Emotion"][completed_samples:])):
                print(f"Processing sample {completed_samples + i + 1}/{total_samples}")
                
                output = llm.generate(prompt)
                reasoning_steps, answer = llm.parse_think_output(output)
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
        total_samples = len(all_prompts["EU"]["Cause"])
        
        if completed_samples < total_samples:
            print(f"Continuing EU-Cause experiments for {llm_name}: {completed_samples}/{total_samples} completed")
            
            for i, (prompt, gt) in enumerate(zip(all_prompts["EU"]["Cause"][completed_samples:], gts["EU"]["Cause"][completed_samples:])):
                print(f"Processing sample {completed_samples + i + 1}/{total_samples}")
                
                output = llm.generate(prompt)
                reasoning_steps, answer = llm.parse_think_output(output)
                sys.stdout.flush()
                
                llm_results["reasoning_steps"].append(reasoning_steps)
                llm_results["answers"].append(answer)
                save_results(results, dataset, lang, cot)
        else:
            print(f"EU-Cause experiments already completed for {llm_name}")
            
        print("\nFinished Cause!\n")
        
    elif dataset == "tombench":
        print("Running TomBench experiment...")
        print(f"TomBench has {len(all_prompts)} categories: {list(all_prompts.keys())}")
        
        if llm_name not in results:
            results[llm_name] = {}
            for category in all_prompts.keys():
                results[llm_name][category] = {
                    "reasoning_steps": [],
                    "answers": []
                }
    
        for category in all_prompts.keys():
            if category not in results[llm_name]:
                results[llm_name][category] = {
                    "reasoning_steps": [],
                    "answers": []
                }
                
            llm_results = results[llm_name][category]
            completed_samples = len(llm_results.get("answers", []))
            total_samples = len(all_prompts[category])
            
            print(f"\nStarting with {category}")
            
            if completed_samples < total_samples:
                print(f"Continuing {category} experiments for {llm_name}: {completed_samples}/{total_samples} completed")
                
                for i, (prompt, gt) in enumerate(zip(all_prompts[category][completed_samples:], gts[category][completed_samples:])):
                    print(f"Processing sample {completed_samples + i + 1}/{total_samples}")
                    
                    output = llm.generate(prompt)
                    reasoning_steps, answer = llm.parse_think_output(output)
                    sys.stdout.flush()
                    
                    llm_results["reasoning_steps"].append(reasoning_steps)
                    llm_results["answers"].append(answer)
                    save_results(results, dataset, lang, cot)
            else:
                print(f"{category} experiments already completed for {llm_name}")
else:
    raise ValueError("Invalid dataset: {}".format(dataset))