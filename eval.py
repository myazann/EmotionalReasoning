import argparse
import sys
import json
import os
from pathlib import Path
from collections import defaultdict

from sklearn.metrics import f1_score
from utils import load_existing_results
from exp_datasets import get_gts, get_emo_eu_cats, get_emo_eu_cat_dict, get_emo_ea_problems_and_relationships, get_dataset, get_tom_abilities

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
        
        flat_gts = []
        flat_preds = []
        
        all_samples_with_abilities = []
        sample_index = 0
        
        tom_data = get_dataset("tombench")
        
        for topic in results[llm_name]:
            if len(gts[topic]) == len(results[llm_name][topic]["answers"]):
                gt_topic = gts[topic]
                answer_topic = results[llm_name][topic]["answers"]
                parsed_answers = [answer.split("\n\n")[-1].strip("[]") for answer in answer_topic]
                
                flat_gts.extend(gt_topic)
                flat_preds.extend(parsed_answers)
                
                topic_data = tom_data[topic]
                for i, (gt, pred) in enumerate(zip(gt_topic, parsed_answers)):
                    if i < len(topic_data):
                        ability = topic_data[i]["能力\nABILITY"]
                        all_samples_with_abilities.append((gt, pred, ability))
                    else:
                        print(f"Warning: Couldn't match ability for sample {i} in topic {topic}")
                
                correct = sum(1 for gt, pred in zip(gt_topic, parsed_answers) if gt == pred)
                accuracy = correct / len(gt_topic) if len(gt_topic) > 0 else 0
                
                f1 = f1_score(gt_topic, parsed_answers, average="macro")
                
                topic_res[topic] = {
                    "accuracy": round(accuracy, 3), 
                    "f1": round(f1, 3),
                    "sample_count": len(gt_topic)
                }
            else:
                print(f"{topic} not completed for {llm_name}")
        
        ability_results = defaultdict(list)
        main_ability_metrics = {}
        
        for gt, pred, ability in all_samples_with_abilities:
            ability_results[ability].append((gt, pred))
            
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
                
                # Merge the two desires influence sub-abilities
                if sub_ability == "desires influence on actions" or sub_ability == "desires influence on emotions (beliefs)":
                    sub_ability = "desires influence on actions and emotions"
            else:
                main_ability = ability.strip().lower()
                sub_ability = "general"
            
            if main_ability not in main_ability_metrics:
                main_ability_metrics[main_ability] = {
                    "samples": [],
                    "sub_abilities": {}
                }
            
            main_ability_metrics[main_ability]["samples"].append((gt, pred))
            
            if sub_ability not in main_ability_metrics[main_ability]["sub_abilities"]:
                main_ability_metrics[main_ability]["sub_abilities"][sub_ability] = []
            
            main_ability_metrics[main_ability]["sub_abilities"][sub_ability].append((gt, pred))
        
        abilities_output = {}
        
        for main_ability, data in main_ability_metrics.items():
            main_samples = data["samples"]
            main_gt = [pair[0] for pair in main_samples]
            main_pred = [pair[1] for pair in main_samples]
            
            correct = sum(1 for gt, pred in main_samples if gt == pred)
            accuracy = correct / len(main_samples) if len(main_samples) > 0 else 0
            
            try:
                f1 = f1_score(main_gt, main_pred, average="macro")
            except:
                f1 = 0
            
            abilities_output[main_ability] = {
                "accuracy": round(accuracy, 3),
                "f1": round(f1, 3),
                "sample_count": len(main_samples),
                "sub_abilities": {}
            }
            
            for sub_ability, sub_samples in data["sub_abilities"].items():
                sub_gt = [pair[0] for pair in sub_samples]
                sub_pred = [pair[1] for pair in sub_samples]
                
                correct = sum(1 for gt, pred in sub_samples if gt == pred)
                accuracy = correct / len(sub_samples) if len(sub_samples) > 0 else 0
                
                try:
                    f1 = f1_score(sub_gt, sub_pred, average="macro")
                except:
                    f1 = 0
                
                abilities_output[main_ability]["sub_abilities"][sub_ability] = {
                    "accuracy": round(accuracy, 3),
                    "f1": round(f1, 3),
                    "sample_count": len(sub_samples)
                }
        
        overall_correct = sum(1 for gt, pred in zip(flat_gts, flat_preds) if gt == pred)
        overall_accuracy = overall_correct / len(flat_gts) if len(flat_gts) > 0 else 0
        
        try:
            overall_f1 = f1_score(flat_gts, flat_preds, average="weighted")
        except:
            overall_f1 = 0
        
        eval_res[llm_name] = {
            "topics": topic_res,
            "abilities": abilities_output,
            "overall": {
                "accuracy": round(overall_accuracy, 3),
                "f1": round(overall_f1, 3),
                "sample_count": len(flat_gts)
            }
        }
        
elif dataset == "emobench":
    eu_categories = get_emo_eu_cats()
    eu_cat_dict = get_emo_eu_cat_dict()
    ea_problems, ea_relationships = get_emo_ea_problems_and_relationships()
    
    for llm_name in results:
        eval_res[llm_name] = {
            "EA": {
                "overall": {},
                "problems": {},
                "relationships": {},
                "problem_relationship_pairs": {}
            },
            "EU": {
                "Emotion": {
                    "overall": {},
                },
                "Cause": {
                    "overall": {},
                }
            }
        }
        
        all_accuracy = []
        all_f1 = []
        all_samples = 0
        
        EA_results = results[llm_name]["EA"]
        if len(EA_results["answers"]) == len(gts["EA"]):
            gt_EA = [str(gt) for gt in gts["EA"]]
            answer_EA = EA_results["answers"]
            parsed_answers = [answer.strip()[-1] for answer in answer_EA]
            
            correct = sum(1 for gt, pred in zip(gt_EA, parsed_answers) if gt == pred)
            accuracy = correct / len(gt_EA) if len(gt_EA) > 0 else 0
            
            f1 = f1_score(gt_EA, parsed_answers, average="macro")
            
            eval_res[llm_name]["EA"]["overall"] = {
                "accuracy": round(accuracy, 3), 
                "f1": round(f1, 3),
                "sample_count": len(gt_EA)
            }
            
            all_accuracy.append((accuracy, len(gt_EA)))
            all_f1.append((f1, len(gt_EA)))
            all_samples += len(gt_EA)
            
            pair_results = defaultdict(list)
            for i, (problem, relationship) in enumerate(zip(ea_problems, ea_relationships)):
                pair_key = f"{problem}:{relationship}"
                pair_results[pair_key].append((gt_EA[i], parsed_answers[i]))
            
            eval_res[llm_name]["EA"]["problem_relationship_pairs"] = {}
            eval_res[llm_name]["EA"]["problems"] = {}
            eval_res[llm_name]["EA"]["relationships"] = {}
            
            problem_results = defaultdict(list)
            relationship_results = defaultdict(list)
            
            for pair_key, result_pairs in pair_results.items():
                problem, relationship = pair_key.split(":")
                
                for gt, pred in result_pairs:
                    problem_results[problem].append((gt, pred))
                    relationship_results[relationship].append((gt, pred))
                
                pair_gt = [pair[0] for pair in result_pairs]
                pair_pred = [pair[1] for pair in result_pairs]
                
                correct = sum(1 for gt, pred in result_pairs if gt == pred)
                accuracy = correct / len(result_pairs) if len(result_pairs) > 0 else 0
                
                try:
                    f1 = f1_score(pair_gt, pair_pred, average="macro")
                except:
                    f1 = 0
                
                eval_res[llm_name]["EA"]["problem_relationship_pairs"][pair_key] = {
                    "accuracy": round(accuracy, 3),
                    "f1": round(f1, 3),
                    "sample_count": len(result_pairs)
                }
            
            for problem, result_pairs in problem_results.items():
                problem_gt = [pair[0] for pair in result_pairs]
                problem_pred = [pair[1] for pair in result_pairs]
                
                correct = sum(1 for gt, pred in result_pairs if gt == pred)
                accuracy = correct / len(result_pairs) if len(result_pairs) > 0 else 0
                
                try:
                    f1 = f1_score(problem_gt, problem_pred, average="macro")
                except:
                    f1 = 0
                
                eval_res[llm_name]["EA"]["problems"][problem] = {
                    "accuracy": round(accuracy, 3),
                    "f1": round(f1, 3),
                    "sample_count": len(result_pairs)
                }
            
            for relationship, result_pairs in relationship_results.items():
                rel_gt = [pair[0] for pair in result_pairs]
                rel_pred = [pair[1] for pair in result_pairs]
                
                correct = sum(1 for gt, pred in result_pairs if gt == pred)
                accuracy = correct / len(result_pairs) if len(result_pairs) > 0 else 0
                
                try:
                    f1 = f1_score(rel_gt, rel_pred, average="macro")
                except:
                    f1 = 0
                
                eval_res[llm_name]["EA"]["relationships"][relationship] = {
                    "accuracy": round(accuracy, 3),
                    "f1": round(f1, 3),
                    "sample_count": len(result_pairs)
                }
        else:
            print(f"EA not completed for {llm_name}")
        
        EU_cause_results = results[llm_name]["EU"]["Cause"]
        if len(EU_cause_results["answers"]) == len(gts["EU"]["Cause"]):
            gt_cause = [str(gt) for gt in gts["EU"]["Cause"]]
            answer_cause = EU_cause_results["answers"]
            parsed_answers = [answer.strip()[-1] for answer in answer_cause]
            
            correct = sum(1 for gt, pred in zip(gt_cause, parsed_answers) if gt == pred)
            accuracy = correct / len(gt_cause) if len(gt_cause) > 0 else 0
            
            f1 = f1_score(gt_cause, parsed_answers, average="macro")
            
            eval_res[llm_name]["EU"]["Cause"]["overall"] = {
                "accuracy": round(accuracy, 3), 
                "f1": round(f1, 3),
                "sample_count": len(gt_cause)
            }
            
            all_accuracy.append((accuracy, len(gt_cause)))
            all_f1.append((f1, len(gt_cause)))
            all_samples += len(gt_cause)
            
            # Map each category to its main category and subcategory
            main_cat_results = defaultdict(list)
            sub_cat_results = defaultdict(list)
            cat_to_main_mapping = {}
            
            # Create mapping of category to main category
            for main_cat, sub_cats in eu_cat_dict.items():
                for sub_cat in sub_cats:
                    cat_to_main_mapping[sub_cat] = main_cat
            
            # Group samples by category, main category, and subcategory
            category_results = defaultdict(list)
            for i, category in enumerate(eu_categories):
                category_results[category].append((gt_cause[i], parsed_answers[i]))
                
                # Add to main category if it exists in the mapping
                if category in cat_to_main_mapping:
                    main_cat = cat_to_main_mapping[category]
                    main_cat_results[main_cat].append((gt_cause[i], parsed_answers[i]))
                    sub_cat_results[category].append((gt_cause[i], parsed_answers[i]))
                else:
                    # If not found in mapping, use the category as its own main category
                    main_cat_results[category].append((gt_cause[i], parsed_answers[i]))
            
            # Calculate metrics for each main category
            for main_cat, result_pairs in main_cat_results.items():
                main_cat_gt = [pair[0] for pair in result_pairs]
                main_cat_pred = [pair[1] for pair in result_pairs]
                
                correct = sum(1 for gt, pred in result_pairs if gt == pred)
                accuracy = correct / len(result_pairs) if len(result_pairs) > 0 else 0
                
                try:
                    f1 = f1_score(main_cat_gt, main_cat_pred, average="macro")
                except:
                    f1 = 0
                
                eval_res[llm_name]["EU"]["Cause"][main_cat] = {
                    "accuracy": round(accuracy, 3),
                    "f1": round(f1, 3),
                    "sample_count": len(result_pairs),
                    "sub_categories": {}
                }
            
            # Calculate metrics for each subcategory and add to its main category
            for sub_cat, result_pairs in sub_cat_results.items():
                if sub_cat in cat_to_main_mapping:
                    main_cat = cat_to_main_mapping[sub_cat]
                    
                    sub_cat_gt = [pair[0] for pair in result_pairs]
                    sub_cat_pred = [pair[1] for pair in result_pairs]
                    
                    correct = sum(1 for gt, pred in result_pairs if gt == pred)
                    accuracy = correct / len(result_pairs) if len(result_pairs) > 0 else 0
                    
                    try:
                        f1 = f1_score(sub_cat_gt, sub_cat_pred, average="macro")
                    except:
                        f1 = 0
                    
                    # Add subcategory to its main category
                    eval_res[llm_name]["EU"]["Cause"][main_cat]["sub_categories"][sub_cat] = {
                        "accuracy": round(accuracy, 3),
                        "f1": round(f1, 3),
                        "sample_count": len(result_pairs)
                    }
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
            
            eval_res[llm_name]["EU"]["Emotion"]["overall"] = {
                "accuracy": round(accuracy, 3), 
                "f1": round(f1, 3),
                "sample_count": len(gt_emotion)
            }
            
            all_accuracy.append((accuracy, len(gt_emotion)))
            all_f1.append((f1, len(gt_emotion)))
            all_samples += len(gt_emotion)
            
            # Map each category to its main category and subcategory
            main_cat_results = defaultdict(list)
            sub_cat_results = defaultdict(list)
            cat_to_main_mapping = {}
            
            # Create mapping of category to main category
            for main_cat, sub_cats in eu_cat_dict.items():
                for sub_cat in sub_cats:
                    cat_to_main_mapping[sub_cat] = main_cat
            
            # Group samples by category, main category, and subcategory
            category_results = defaultdict(list)
            for i, category in enumerate(eu_categories):
                category_results[category].append((gt_emotion[i], parsed_answers[i]))
                
                # Add to main category if it exists in the mapping
                if category in cat_to_main_mapping:
                    main_cat = cat_to_main_mapping[category]
                    main_cat_results[main_cat].append((gt_emotion[i], parsed_answers[i]))
                    sub_cat_results[category].append((gt_emotion[i], parsed_answers[i]))
                else:
                    # If not found in mapping, use the category as its own main category
                    main_cat_results[category].append((gt_emotion[i], parsed_answers[i]))
                        
            # Calculate metrics for each main category
            for main_cat, result_pairs in main_cat_results.items():
                main_cat_gt = [pair[0] for pair in result_pairs]
                main_cat_pred = [pair[1] for pair in result_pairs]
                
                correct = sum(1 for gt, pred in result_pairs if gt == pred)
                accuracy = correct / len(result_pairs) if len(result_pairs) > 0 else 0
                
                try:
                    f1 = f1_score(main_cat_gt, main_cat_pred, average="macro")
                except:
                    f1 = 0
                
                eval_res[llm_name]["EU"]["Emotion"][main_cat] = {
                    "accuracy": round(accuracy, 3),
                    "f1": round(f1, 3),
                    "sample_count": len(result_pairs),
                    "sub_categories": {}
                }
            
            # Calculate metrics for each subcategory and add to its main category
            for sub_cat, result_pairs in sub_cat_results.items():
                if sub_cat in cat_to_main_mapping:
                    main_cat = cat_to_main_mapping[sub_cat]
                    
                    sub_cat_gt = [pair[0] for pair in result_pairs]
                    sub_cat_pred = [pair[1] for pair in result_pairs]
                    
                    correct = sum(1 for gt, pred in result_pairs if gt == pred)
                    accuracy = correct / len(result_pairs) if len(result_pairs) > 0 else 0
                    
                    try:
                        f1 = f1_score(sub_cat_gt, sub_cat_pred, average="macro")
                    except:
                        f1 = 0
                    
                    # Add subcategory to its main category
                    eval_res[llm_name]["EU"]["Emotion"][main_cat]["sub_categories"][sub_cat] = {
                        "accuracy": round(accuracy, 3),
                        "f1": round(f1, 3),
                        "sample_count": len(result_pairs)
                    }
        else:
            print(f"Emotion not completed for {llm_name}")
        
        def has_complete_metrics(result_dict, task):
            return ("overall" in result_dict[task] and 
                    isinstance(result_dict[task]["overall"], dict) and
                    "accuracy" in result_dict[task]["overall"] and 
                    "f1" in result_dict[task]["overall"] and
                    "sample_count" in result_dict[task]["overall"])
        
        has_emotion_metrics = has_complete_metrics(eval_res[llm_name]["EU"], "Emotion")
        has_cause_metrics = has_complete_metrics(eval_res[llm_name]["EU"], "Cause")
        
        if has_emotion_metrics and has_cause_metrics:
            emotion_acc = eval_res[llm_name]["EU"]["Emotion"]["overall"]["accuracy"]
            emotion_f1 = eval_res[llm_name]["EU"]["Emotion"]["overall"]["f1"]
            emotion_count = eval_res[llm_name]["EU"]["Emotion"]["overall"]["sample_count"]
            
            cause_acc = eval_res[llm_name]["EU"]["Cause"]["overall"]["accuracy"]
            cause_f1 = eval_res[llm_name]["EU"]["Cause"]["overall"]["f1"]
            cause_count = eval_res[llm_name]["EU"]["Cause"]["overall"]["sample_count"]
            
            total_count = emotion_count + cause_count
            overall_eu_acc = ((emotion_acc * emotion_count) + (cause_acc * cause_count)) / total_count if total_count > 0 else 0
            overall_eu_f1 = ((emotion_f1 * emotion_count) + (cause_f1 * cause_count)) / total_count if total_count > 0 else 0
            
            eval_res[llm_name]["EU"]["overall"] = {
                "accuracy": round(overall_eu_acc, 3),
                "f1": round(overall_eu_f1, 3),
                "sample_count": total_count
            }

        

            
        print(f"\nFinished {llm_name}!\n")
        
with open(os.path.join("results", f"{dataset}_eval_res.json"), "w") as f:
    json.dump(eval_res, f, indent=4)
