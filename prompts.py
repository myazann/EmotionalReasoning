import os 
import json

def get_all_prompts(dataset, data=None, lang="en", cot=False):
    """
    Generate prompts for the specified dataset using provided data.
    
    Args:
        dataset (str): Name of the dataset, either 'emobench' or 'tombench'.
        data (dict): Dataset samples. If None, the function will return an empty result.
        lang (str): Language code, default is 'en'.
        cot (bool): Whether to generate chain-of-thought prompts, default is False.
        
    Returns:
        dict: Dictionary of prompts organized by dataset structure.
    """
    if dataset == "emobench":
        if data is None:
            return {"EA": [], "EU": {"Emotion": [], "Cause": []}}
            
        ea_data = data.get("EA", [])
        eu_data = data.get("EU", [])

        all_prompts = {
            "EA": [],
            "EU": {
                "Emotion": [],
                "Cause": []
            }
        }

        for ea_sample in ea_data:
            ea_prompt_params = {
                "scenario": ea_sample["Scenario"][lang],
                "q_type": ea_sample["Question"][lang],
                "choices": ea_sample["Choices"][lang],
                "subject": ea_sample["Subject"][lang]
            }
            EA_prompt = construct_emobench_prompt(task="EA", lang=lang, cot=cot, prompt_params=ea_prompt_params)
            all_prompts["EA"].append(EA_prompt)

        for eu_sample in eu_data:
            eu_prompt_emotion_params = {
                "scenario": eu_sample["Scenario"][lang],
                "choices": eu_sample["Emotion"]["Choices"][lang],
                "subject": eu_sample["Subject"][lang]
            }
            EU_prompt_emotion = construct_emobench_prompt(task="EU", eu_task="Emotion", lang=lang, cot=cot, 
                                        prompt_params=eu_prompt_emotion_params)
            

            eu_prompt_cause_params = eu_prompt_emotion_params.copy()
            eu_prompt_cause_params["choices"] = eu_sample["Cause"]["Choices"][lang]
            eu_prompt_cause_params["emotions"] = eu_sample["Emotion"]["Label"][lang]
            EU_prompt_cause = construct_emobench_prompt(task="EU", eu_task="Cause", lang=lang, cot=cot, 
                                prompt_params=eu_prompt_cause_params)

            all_prompts["EU"]["Emotion"].append(EU_prompt_emotion)
            all_prompts["EU"]["Cause"].append(EU_prompt_cause)

    elif dataset == "tombench":
        all_prompts = {}

        for scenario in data:
            all_prompts[scenario] = []
            for sample in data[scenario]:
                prompt_params = {
                    "story": sample["STORY"],
                    "question": sample["QUESTION"],
                    "choice_a": sample["OPTION-A"],
                    "choice_b": sample["OPTION-B"],
                }
                if "OPTION-C" in sample.keys():
                    prompt_params["choice_c"] = sample["OPTION-C"]
                    prompt_params["choice_d"] = sample["OPTION-D"]

                prompt = construct_tombench_prompt(prompt_params, lang=lang, cot=cot)
                all_prompts[scenario].append(prompt)
    else:
        raise ValueError(f"Dataset {dataset} not defined")
        
    return all_prompts

def construct_tombench_prompt(prompt_params, lang="en", cot=False):
    
    with open(os.path.join("datasets", "ToMBench", "prompts.json"), "r") as f:
        prompts = json.load(f)
    
    cot = "cot" if cot else ""
    sys_prompt_name = f"SystemEvaluatePrompt_{lang}"
    if cot:
        sys_prompt_name += f"_{cot}"
    
    if "choice_c" in prompt_params:
        user_prompt_name = f"UserEvaluatePrompt4Choices_{lang}"
    else:
        user_prompt_name = f"UserEvaluatePrompt2Choices_{lang}"
    
    sys_prompt = prompts[sys_prompt_name]
    user_prompt = prompts[user_prompt_name]
    
    prompt = [{
        "role": "system",
        "content": sys_prompt
    },
    {
        "role": "user",
        "content": user_prompt.format(**prompt_params)
    }]
    return prompt

def construct_emobench_prompt(prompt_params, task, eu_task="Emotion", lang="en", cot=False):

    with open(os.path.join("datasets", "EmoBench", "data", "dicts.json"), "r") as f:
        prompts = json.load(f)
    
    cot = "cot" if cot else "no_cot"
    sys_prompt = prompts["Prompts"]["System"][lang]
    sys_prompt += "The options are numbered starting from 0, with the leftmost option being the first, and so on. Keep in mind that the letter means a number.\n"
    if task == "EU":
        if eu_task not in ["Cause", "Emotion"]:
            raise ValueError(f"EU task {eu_task} not defined")
        task_prompt = prompts["Prompts"]["EU"][eu_task][lang]
    elif task == "EA":
        task_prompt = prompts["Prompts"]["EA"][lang]
    else:
        raise ValueError(f"Task {task} not defined")
    
    cot_prompt = prompts["Prompts"][cot][lang]

    prompt = [{
        "role": "system",
        "content": sys_prompt
    },
    {
        "role": "user",
        "content": task_prompt.format(**prompt_params) + cot_prompt
    }]
    return prompt