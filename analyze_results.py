import json

from exp_datasets import get_gts

eval_res = {}
for file in ["tombench_results.json", "emobench_results.json"]:
    with open(f"results/{file}", "r") as f:
        data = json.load(f)
    eval_res[file.split("_")[0]] = data

results = []
for file in ["tombench_en_generations.json", "emobench_en_generations.json"]:
    with open(f"results/{file}", "r") as f:
        data = json.load(f)
    results.append(data)

models = list(eval_res["emobench"].keys())

tom_gts = get_gts("tombench")
emo_gts = get_gts("emobench")

sample = results[0][models[0]][0]
topic = sample["topic"]
print(sample["answer"])
print(tom_gts[topic])
