import argparse

from prompts import get_all_prompts
from exp_datasets import get_dataset, get_gts
from models import LLM

parser = argparse.ArgumentParser(description='Run experiment with language model')
parser.add_argument('--lang', type=str, choices=['en', 'zh'], default='en',
                    help='Language for the experiment (en or zh)')
parser.add_argument('--cot', action='store_true', default=False,
                    help='Whether to use chain-of-thought prompting')
parser.add_argument('--dataset', type=str, choices=['emobench', 'tombench'], default='emobench',
                    help='Dataset to use (emobench or tombench)')

args = parser.parse_args()
lang = args.lang
cot = args.cot
dataset = args.dataset

# Get the dataset and prompts
dataset_data = get_dataset(dataset)
all_prompts = get_all_prompts(dataset, data=dataset_data, lang=lang, cot=cot)
gts = get_gts(dataset)

# Initialize LLM
llm = LLM("GPT-4o-mini")

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
        

    # print("EU has two subcategories: Emotion and Cause")
        