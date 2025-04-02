import argparse
import sys

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

dataset_data = get_dataset(dataset)
all_prompts = get_all_prompts(dataset, data=dataset_data, lang=lang, cot=cot)
gts = get_gts(dataset)

llm = LLM("QWEN-QwQ-32B-GGUF")
# llm = LLM("DEEPSEEK-R1")

exp_len = 10

if dataset == "emobench":
    print("Running EmoBench experiment...")
    print("Emobench has two categories: EA and EU")
    print("Starting with EA: ")

    for prompt, gt in zip(all_prompts["EA"][:exp_len], gts["EA"][:exp_len]):
        print(prompt)
        print("---")
        answer = llm.generate(prompt)
        print(answer)
        print("---")
        print(gt)
        print("---")
        sys.stdout.flush()

    # print("EU has two subcategories: Emotion and Cause")
elif dataset == "tombench":
    print("Running TomBench experiment...")
    print("TomBench has {} categories: {}".format(len(all_prompts), list(all_prompts.keys())))

    for category in all_prompts.keys():
        print("\nStarting with {}".format(category))
        for prompt, gt in zip(all_prompts[category][:exp_len], gts[category][:exp_len]):
            print(prompt)
            print("---")
            response = llm.generate(prompt)
            answer = response.strip("[]").strip()
            print(answer)
            print("---")
            print(gt)
            print("---")
            sys.stdout.flush()

else:
    raise ValueError("Invalid dataset: {}".format(dataset))