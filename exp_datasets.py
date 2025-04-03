import os
import json

def get_tom_data():
    """
    Gets the ToMBench dataset samples.
    
    Returns:
        dict: A dictionary mapping scenario names to lists of samples.
    """
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
    """
    Gets the EmoBench dataset samples.
    
    Returns:
        dict: A dictionary containing 'EA' and 'EU' data.
    """
    emo_data = {
        "EA": None,
        "EU": None
    }
    
    # Load EA data
    ea_path = os.path.join("datasets", "EmoBench", "data", "EA", "data.json")
    if os.path.exists(ea_path):
        with open(ea_path, "r") as f:
            emo_data["EA"] = json.load(f)
    
    # Load EU data
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

def get_emo_ea_problems():

    emo_data = get_emo_data()
    problems = []
    for sample in emo_data["EA"]:
        problems.append(sample["Problem"])
    
    return problems

def get_emo_ea_relationships():

    emo_data = get_emo_data()
    relationships = []
    for sample in emo_data["EA"]:
        relationships.append(sample["Relationship"])
    
    return relationships

def get_emo_eu_cats():

    cats = []
    emo_data = get_emo_data()

    for sample in emo_data["EU"]:
        cats.append(sample["Category"])
    
    return cats

def get_gts(dataset_name):
    """
    Gets ground truth labels for the specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset, either 'emobench' or 'tombench'.
        
    Returns:
        dict: Ground truth labels.
        
    Raises:
        ValueError: If dataset_name is not recognized.
    """
    if dataset_name.lower() == "emobench":
        return get_emo_gts()
    elif dataset_name.lower() == "tombench":
        return get_tom_gts()
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized. Use 'emobench' or 'tombench'.")

def get_dataset(dataset_name):
    """
    Gets data for the specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset, either 'emobench' or 'tombench'.
        
    Returns:
        dict: Dataset samples.
        
    Raises:
        ValueError: If dataset_name is not recognized.
    """
    if dataset_name.lower() == "emobench":
        return get_emo_data()
    elif dataset_name.lower() == "tombench":
        return get_tom_data()
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized. Use 'emobench' or 'tombench'.")


def get_gts(dataset_name):
    if dataset_name.lower() == "emobench":
        return get_emo_gts()
    elif dataset_name.lower() == "tombench":
        return get_tom_gts()
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized. Use 'emobench' or 'tombench'.")