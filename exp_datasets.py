import os
import json

def get_tom_data():

    tom_data_path = os.path.join("datasets", "ToMBench", "data")
    
    all_scenarios = get_tom_scenarios()
    
    tombench_data = {}
    for scenario in all_scenarios:
        if scenario.endswith(".jsonl"):
            data = []
            with open(os.path.join(tom_data_path, scenario), "r") as f:
                for line in f:
                    data.append(json.loads(line))
            tombench_data[scenario.split(".")[0]] = data
    
    return tombench_data

def get_tom_scenarios():
    
    tom_data_path = os.path.join("datasets", "ToMBench", "data")
    all_scenarios = [f for f in os.listdir(tom_data_path) if f.endswith(".jsonl")]
    
    return all_scenarios

def get_tom_abilities():
    
    dataset = get_tom_data()
    
    all_abilities = []
    for scenario in dataset:
        scenario_data = dataset[scenario]
        for sample in scenario_data:
            all_abilities.append(sample["能力\nABILITY"])
    
    return all_abilities

def get_tom_gts():
    
    all_scenarios = get_tom_scenarios()

    tombench_gts = {}
    tom_data = get_tom_data()

    for scenario in all_scenarios:
        scenario_name = scenario.split(".")[0]
        scenario_data = tom_data[scenario_name]   
        tombench_gts[scenario_name] = []

        for sample in scenario_data:
            tombench_gts[scenario_name].append(sample["答案\nANSWER"])
            
    return tombench_gts

def get_emo_data():

    emo_data = {
        "EA": None,
        "EU": None
    }
    
    ea_path = os.path.join("datasets", "EmoBench", "data", "EA", "data.json")
    if os.path.exists(ea_path):
        with open(ea_path, "r") as f:
            emo_data["EA"] = json.load(f)
    
    eu_path = os.path.join("datasets", "EmoBench", "data", "EU", "data.json")
    if os.path.exists(eu_path):
        with open(eu_path, "r") as f:
            emo_data["EU"] = json.load(f)
    
    return emo_data

def get_emo_gts(lang="en"):

    emo_data = get_emo_data()
    emo_gts = {
        "EA": [],
        "EU": {
            "Emotion": [],
            "Cause": []
        }
    }
    
    for sample in emo_data["EA"]:
        emo_gts["EA"].append(sample["Label"])
    
    for sample in emo_data["EU"]:
        options = sample["Emotion"]["Choices"][lang]
        label_idx = options.index(sample["Emotion"]["Label"][lang])
        emo_gts["EU"]["Emotion"].append(label_idx)
        options = sample["Cause"]["Choices"][lang]
        label_idx = options.index(sample["Cause"]["Label"][lang])
        emo_gts["EU"]["Cause"].append(label_idx)
    
    return emo_gts

def get_emo_ea_problems_and_relationships():

    emo_data = get_emo_data()
    problems = []
    relationships = []
    for sample in emo_data["EA"]:
        problems.append(sample["Problem"])
        relationships.append(sample["Relationship"])
    
    return problems, relationships

def get_emo_eu_cat_dict():
    return {
        "complex_emotions": {
            "emotion_transition",
            "mixture_of_emotions",
            "unexpected_outcome"
        },
        "personal_beliefs_and_experiences": {
            "cultural_value",
            "sentimental_value",
            "persona"
        },
        "emotional_cues": {
            "vocal_cues",
            "visual_cues"
        },
        "perspective_taking": {
            "faux_pas",
            "strange_story",
            "false_belief"
        }
    }

def get_emo_eu_cats():

    cats = []
    emo_data = get_emo_data()
    for sample in emo_data["EU"]:
        cats.append(sample["Category"])
    
    return cats

def get_gts(dataset_name):

    if dataset_name.lower() == "emobench":
        return get_emo_gts()
    elif dataset_name.lower() == "tombench":
        return get_tom_gts()
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized. Use 'emobench' or 'tombench'.")

def get_dataset(dataset_name):

    if dataset_name.lower() == "emobench":
        return get_emo_data()
    elif dataset_name.lower() == "tombench":
        return get_tom_data()
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized. Use 'emobench' or 'tombench'.")