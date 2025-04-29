import argparse
import json
from pathlib import Path
from collections import defaultdict

from utils import load_existing_results
from exp_datasets import get_gts

parser = argparse.ArgumentParser(description='Evaluate LLM performance')
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

# Load ground truth answers
gts = get_gts(dataset)
# Load restructured results directly
results = load_existing_results(dataset, lang, cot)

eval_res = {}

if dataset == "tombench":
    # Process tombench results
    for llm_name, model_samples in results.items():
        print(f"Evaluating {llm_name} on tombench dataset...")
        eval_res[llm_name] = {}
        
        # Create containers for different metrics
        topic_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        ability_metrics = defaultdict(lambda: {"correct": 0, "total": 0, "sub_abilities": defaultdict(lambda: {"correct": 0, "total": 0})})
        overall_correct = 0
        overall_total = 0
        
        # Group samples by topic
        samples_by_topic = defaultdict(list)
        for sample in model_samples:
            topic = sample["topic"]
            samples_by_topic[topic].append(sample)
        
        # Evaluate each topic separately to maintain the correct order
        for topic, topic_samples in samples_by_topic.items():
                
            # Make sure we don't process more samples than we have ground truth for
            gt_list = gts[topic]
            # Sort samples to ensure consistent order
            sorted_samples = sorted(topic_samples, key=lambda x: model_samples.index(x))
            
            # Process only up to the number of ground truth samples
            for i, (sample, gt) in enumerate(zip(sorted_samples[:len(gt_list)], gt_list)):
                main_ability = sample["main_ability"]
                sub_ability = sample["sub_ability"]
                answer = sample["answer"]
                
                # Extract the predicted answer
                parsed_answer = answer.split("\n\n")[-1].strip("[]")
                
                # Check if the answer is correct
                is_correct = (gt == parsed_answer)
                
                # Update metrics
                if is_correct:
                    topic_metrics[topic]["correct"] += 1
                    ability_metrics[main_ability]["correct"] += 1
                    ability_metrics[main_ability]["sub_abilities"][sub_ability]["correct"] += 1
                    overall_correct += 1
                
                # Update totals
                topic_metrics[topic]["total"] += 1
                ability_metrics[main_ability]["total"] += 1
                ability_metrics[main_ability]["sub_abilities"][sub_ability]["total"] += 1
                overall_total += 1
            
            # Report progress
            print(f"  Processed {len(gt_list)} samples for topic {topic}")
        
        # Calculate accuracies and create output structure
        topic_res = {}
        for topic, metrics in topic_metrics.items():
            accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
            topic_res[topic] = {
                "accuracy": round(accuracy, 3),
                "sample_count": metrics["total"]
            }
        
        abilities_output = {}
        for main_ability, metrics in ability_metrics.items():
            accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
            abilities_output[main_ability] = {
                "accuracy": round(accuracy, 3),
                "sample_count": metrics["total"],
                "sub_abilities": {}
            }
            
            for sub_ability, sub_metrics in metrics["sub_abilities"].items():
                sub_accuracy = sub_metrics["correct"] / sub_metrics["total"] if sub_metrics["total"] > 0 else 0
                abilities_output[main_ability]["sub_abilities"][sub_ability] = {
                    "accuracy": round(sub_accuracy, 3),
                    "sample_count": sub_metrics["total"]
                }
        
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
        
        eval_res[llm_name] = {
            "topics": topic_res,
            "abilities": abilities_output,
            "overall": {
                "accuracy": round(overall_accuracy, 3),
                "sample_count": overall_total
            }
        }

elif dataset == "emobench":
    # Process emobench results
    for llm_name, model_data in results.items():
        print(f"Evaluating {llm_name} on emobench dataset...")
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
                    "categories": {}
                },
                "Cause": {
                    "overall": {},
                    "categories": {}
                }
            }
        }
        
        all_accuracy = []
        all_samples = 0
        
        # Process EA samples
        if "EA" in model_data and isinstance(model_data["EA"], list):
            ea_samples = model_data["EA"]
            gt_EA = [str(gt) for gt in gts["EA"]]
            
            # Check if we have the right number of samples
            if len(ea_samples) == len(gt_EA):
                # Extract and parse EA answers
                parsed_answers = [sample["answer"].strip()[-1] for sample in ea_samples]
                
                # Calculate overall EA accuracy
                correct = sum(1 for gt, pred in zip(gt_EA, parsed_answers) if gt == pred)
                accuracy = correct / len(gt_EA) if len(gt_EA) > 0 else 0
                
                eval_res[llm_name]["EA"]["overall"] = {
                    "accuracy": round(accuracy, 3),
                    "sample_count": len(gt_EA)
                }
                
                all_accuracy.append((accuracy, len(gt_EA)))
                all_samples += len(gt_EA)
                
                # Prepare structures for problem/relationship analysis
                pair_results = defaultdict(list)
                problem_results = defaultdict(list)
                relationship_results = defaultdict(list)
                
                # Process each sample to extract problems and relationships
                for i, sample in enumerate(ea_samples):
                    problem = sample["problem"]
                    relationship = sample["relationship"]
                    pair_key = f"{problem}:{relationship}"
                    
                    # Store results by problem/relationship pairs
                    pair_results[pair_key].append((gt_EA[i], parsed_answers[i]))
                    problem_results[problem].append((gt_EA[i], parsed_answers[i]))
                    relationship_results[relationship].append((gt_EA[i], parsed_answers[i]))
                
                # Calculate metrics for problem-relationship pairs
                for pair, results_list in pair_results.items():
                    correct = sum(1 for gt, pred in results_list if gt == pred)
                    accuracy = correct / len(results_list) if len(results_list) > 0 else 0
                    
                    eval_res[llm_name]["EA"]["problem_relationship_pairs"][pair] = {
                        "accuracy": round(accuracy, 3),
                        "sample_count": len(results_list)
                    }
                
                # Calculate metrics for problems
                for problem, results_list in problem_results.items():
                    correct = sum(1 for gt, pred in results_list if gt == pred)
                    accuracy = correct / len(results_list) if len(results_list) > 0 else 0
                    
                    eval_res[llm_name]["EA"]["problems"][problem] = {
                        "accuracy": round(accuracy, 3),
                        "sample_count": len(results_list)
                    }
                
                # Calculate metrics for relationships
                for relationship, results_list in relationship_results.items():
                    correct = sum(1 for gt, pred in results_list if gt == pred)
                    accuracy = correct / len(results_list) if len(results_list) > 0 else 0
                    
                    eval_res[llm_name]["EA"]["relationships"][relationship] = {
                        "accuracy": round(accuracy, 3),
                        "sample_count": len(results_list)
                    }
            else:
                print(f"Warning: Number of EA samples ({len(ea_samples)}) does not match ground truth ({len(gt_EA)})")
        
        # Process EU Cause samples
        if "EU" in model_data and "Cause" in model_data["EU"] and isinstance(model_data["EU"]["Cause"], list):
            eu_cause_samples = model_data["EU"]["Cause"]
            gt_cause = [str(gt) for gt in gts["EU"]["Cause"]]
            
            # Check if we have the right number of samples
            if len(eu_cause_samples) == len(gt_cause):
                # Extract and parse answers
                parsed_answers = []
                for sample in eu_cause_samples:
                    answer = sample["answer"]
                    answer_text = answer.strip()
                    
                    # Handle empty answers
                    if not answer_text:
                        parsed_answers.append("")  # Empty answer
                        continue
                    
                    # Extract final answer based on format (last character or specific pattern)
                    if answer_text[-1].isdigit():
                        parsed_answers.append(answer_text[-1])
                    else:
                        answer_parts = answer_text.split("Answer:")
                        if len(answer_parts) > 1 and answer_parts[1].strip():
                            parsed_answers.append(answer_parts[1].strip()[0])
                        else:
                            # Fallback - use the last character if available, otherwise empty string
                            parsed_answers.append(answer_text[-1] if answer_text else "")
                
                # Calculate overall Cause accuracy
                correct = sum(1 for gt, pred in zip(gt_cause, parsed_answers) if gt == pred)
                accuracy = correct / len(gt_cause) if len(gt_cause) > 0 else 0
                
                eval_res[llm_name]["EU"]["Cause"]["overall"] = {
                    "accuracy": round(accuracy, 3),
                    "sample_count": len(gt_cause)
                }
                
                all_accuracy.append((accuracy, len(gt_cause)))
                all_samples += len(gt_cause)
                
                # Group by category for category-level metrics
                category_results = defaultdict(list)
                
                for i, sample in enumerate(eu_cause_samples):
                    category = sample["category"]
                    category_results[category].append((gt_cause[i], parsed_answers[i]))
                
                # Calculate metrics for each category
                for category, results_list in category_results.items():
                    correct = sum(1 for gt, pred in results_list if gt == pred)
                    accuracy = correct / len(results_list) if len(results_list) > 0 else 0
                    
                    eval_res[llm_name]["EU"]["Cause"]["categories"][category] = {
                        "accuracy": round(accuracy, 3),
                        "sample_count": len(results_list)
                    }
            else:
                print(f"Warning: Number of EU Cause samples ({len(eu_cause_samples)}) does not match ground truth ({len(gt_cause)})")
        
        # Process EU Emotion samples
        if "EU" in model_data and "Emotion" in model_data["EU"] and isinstance(model_data["EU"]["Emotion"], list):
            eu_emotion_samples = model_data["EU"]["Emotion"]
            gt_emotion = [str(gt) for gt in gts["EU"]["Emotion"]]
            
            # Check if we have the right number of samples
            if len(eu_emotion_samples) == len(gt_emotion):
                # Extract and parse answers
                parsed_answers = []
                for sample in eu_emotion_samples:
                    answer = sample["answer"]
                    answer_text = answer.strip()
                    
                    # Handle empty answers
                    if not answer_text:
                        parsed_answers.append("")  # Empty answer
                        continue
                    
                    # Extract final answer based on format (last character or specific pattern)
                    if answer_text[-1].isdigit():
                        parsed_answers.append(answer_text[-1])
                    else:
                        answer_parts = answer_text.split("Answer:")
                        if len(answer_parts) > 1 and answer_parts[1].strip():
                            parsed_answers.append(answer_parts[1].strip()[0])
                        else:
                            # Fallback - use the last character if available, otherwise empty string
                            parsed_answers.append(answer_text[-1] if answer_text else "")
                
                # Calculate overall Emotion accuracy
                correct = sum(1 for gt, pred in zip(gt_emotion, parsed_answers) if gt == pred)
                accuracy = correct / len(gt_emotion) if len(gt_emotion) > 0 else 0
                
                eval_res[llm_name]["EU"]["Emotion"]["overall"] = {
                    "accuracy": round(accuracy, 3),
                    "sample_count": len(gt_emotion)
                }
                
                all_accuracy.append((accuracy, len(gt_emotion)))
                all_samples += len(gt_emotion)
                
                # Group by category for category-level metrics
                category_results = defaultdict(list)
                
                for i, sample in enumerate(eu_emotion_samples):
                    category = sample["category"]
                    category_results[category].append((gt_emotion[i], parsed_answers[i]))
                
                # Calculate metrics for each category
                for category, results_list in category_results.items():
                    correct = sum(1 for gt, pred in results_list if gt == pred)
                    accuracy = correct / len(results_list) if len(results_list) > 0 else 0
                    
                    eval_res[llm_name]["EU"]["Emotion"]["categories"][category] = {
                        "accuracy": round(accuracy, 3),
                        "sample_count": len(results_list)
                    }
            else:
                print(f"Warning: Number of EU Emotion samples ({len(eu_emotion_samples)}) does not match ground truth ({len(gt_emotion)})")
        
        # Calculate overall EU metrics if both Emotion and Cause are present
        if "Emotion" in eval_res[llm_name]["EU"] and "overall" in eval_res[llm_name]["EU"]["Emotion"] and \
           "Cause" in eval_res[llm_name]["EU"] and "overall" in eval_res[llm_name]["EU"]["Cause"]:
            # Calculate weighted average accuracy
            emotion_acc = eval_res[llm_name]["EU"]["Emotion"]["overall"]["accuracy"]
            emotion_count = eval_res[llm_name]["EU"]["Emotion"]["overall"]["sample_count"]
            cause_acc = eval_res[llm_name]["EU"]["Cause"]["overall"]["accuracy"]
            cause_count = eval_res[llm_name]["EU"]["Cause"]["overall"]["sample_count"]
            
            eu_total = emotion_count + cause_count
            eu_acc = (emotion_acc * emotion_count + cause_acc * cause_count) / eu_total if eu_total > 0 else 0
            
            eval_res[llm_name]["EU"]["overall"] = {
                "accuracy": round(eu_acc, 3),
                "sample_count": eu_total
            }
        
        # Calculate overall metrics across EA and EU
        if "EA" in eval_res[llm_name] and "overall" in eval_res[llm_name]["EA"] and \
           "EU" in eval_res[llm_name] and "overall" in eval_res[llm_name]["EU"]:
            # Calculate weighted average accuracy
            total_correct = sum(acc * count for acc, count in all_accuracy)
            overall_acc = total_correct / all_samples if all_samples > 0 else 0
            
            eval_res[llm_name]["overall"] = {
                "accuracy": round(overall_acc, 3),
                "sample_count": all_samples
            }

# Save evaluation results
output_path = results_dir / f"{dataset}_{lang}_results.json"
with open(output_path, "w") as f:
    json.dump(eval_res, f, indent=4)

print(f"Evaluation results saved to {output_path}")

# Print overall results
print("\nOverall Results:")
for llm_name in eval_res:
    if "overall" in eval_res[llm_name]:
        print(f"{llm_name}: {eval_res[llm_name]['overall']['accuracy']:.3f} ({eval_res[llm_name]['overall']['sample_count']} samples)")
    elif dataset == "tombench":
        print(f"{llm_name}: {eval_res[llm_name]['overall']['accuracy']:.3f} ({eval_res[llm_name]['overall']['sample_count']} samples)")
    else:
        print(f"{llm_name}: Results incomplete")
