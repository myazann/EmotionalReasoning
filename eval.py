import argparse
import sys
import json
import os
from pathlib import Path

from sklearn.metrics import f1_score
from utils import load_existing_results
from exp_datasets import get_gts

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

gts = get_gts(dataset)
results = load_existing_results(dataset, lang, cot)

eval_res = {}

if dataset == "tombench":
    for llm_name in results:
        eval_res[llm_name] = {}
        topic_res = {}
        for topic in results[llm_name]:
            if len(gts[topic]) == len(results[llm_name][topic]["answers"]):
                gt_topic = gts[topic]
                answer_topic = results[llm_name][topic]["answers"]
                parsed_answers = [answer.split("\n\n")[-1].strip("[]") for answer in answer_topic]
                
                correct = sum(1 for gt, pred in zip(gt_topic, parsed_answers) if gt == pred)
                accuracy = correct / len(gt_topic) if len(gt_topic) > 0 else 0
                
                f1 = f1_score(gt_topic, parsed_answers, average="macro")
                
                topic_res[topic] = {"accuracy": round(accuracy, 3), "f1": round(f1, 3)}
            else:
                print(f"{topic} not completed for {llm_name}")

        eval_res[llm_name] = topic_res
        
elif dataset == "emobench":
    for llm_name in results:
        eval_res[llm_name] = {
            "EA": {},
            "EU": {
                "Emotion": {},
                "Cause": {}
            }
        }
        EA_results = results[llm_name]["EA"]
        if len(EA_results["answers"]) == len(gts["EA"]):
            
            gt_EA = [str(gt) for gt in gts["EA"]]
            answer_EA = EA_results["answers"]
            parsed_answers = [answer.strip()[-1] for answer in answer_EA]
            
            correct = sum(1 for gt, pred in zip(gt_EA, parsed_answers) if gt == pred)
            accuracy = correct / len(gt_EA) if len(gt_EA) > 0 else 0
            
            f1 = f1_score(gt_EA, parsed_answers, average="macro")
            
            eval_res[llm_name]["EA"] = {"accuracy": round(accuracy, 3), "f1": round(f1, 3)}
        else:
            print(f"EA not completed for {llm_name}")
            
        EU_cause_results = results[llm_name]["EU"]["Cause"]
        if len(EU_cause_results["answers"]) == len(gts["EU"]["Cause"]):
            
            gt_cause =   [str(gt) for gt in gts["EU"]["Cause"]]
            answer_cause = EU_cause_results["answers"]
            parsed_answers = [answer.strip()[-1] for answer in answer_cause]
            
            correct = sum(1 for gt, pred in zip(gt_cause, parsed_answers) if gt == pred)
            accuracy = correct / len(gt_cause) if len(gt_cause) > 0 else 0
            
            f1 = f1_score(gt_cause, parsed_answers, average="macro")
            
            eval_res[llm_name]["EU"]["Cause"] = {"accuracy": round(accuracy, 3), "f1": round(f1, 3)}
        else:
            print(f"Cause not completed for {llm_name}")

        EU_emotion_results = results[llm_name]["EU"]["Emotion"]
        if len(EU_emotion_results["answers"]) == len(gts["EU"]["Emotion"]):
            
            gt_emotion = [str(gt) for gt in gts["EU"]["Emotion"]]
            answer_emotion = EU_emotion_results["answers"]
            parsed_answers = [answer.strip()[-1] for answer in answer_emotion]
            
            correct = sum(1 for gt, pred in zip(gt_emotion, parsed_answers) if gt == pred)
            accuracy = correct / len(gt_emotion) if len(gt_emotion) > 0 else 0
            
            f1 = f1_score(gt_emotion, parsed_answers, average="macro")
            
            eval_res[llm_name]["EU"]["Emotion"] = {"accuracy": round(accuracy, 3), "f1": round(f1, 3)}
        else:
            print(f"Emotion not completed for {llm_name}")
            
        print(f"\nFinished {llm_name}!\n")
        
with open(os.path.join("results", f"{dataset}_eval_res.json"), "w") as f:
    json.dump(eval_res, f, indent=4)
