import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Data:
    prompt: str
    metric: str
    key: str = 'text;caption;captions;label;labels'

DATASET_PROMPTS = {
    "VocalSound": Data(
        prompt="Classify the vocal sound.",
        metric="Accuracy",  
    ),
    "UrbanSound8k": Data(
        prompt="Classify the urban sound.",
        metric="Accuracy",
    ),
    "GTZAN": Data(
        prompt="Classify the music genre (GTZAN).",
        metric="Accuracy",
    ),
    "FreeMusicArchive": Data(
        prompt="Classify the music genre (FMA).",
        metric="Accuracy",
    ),
    "NSynth": Data(
        prompt="Classify the music instrument.",
        metric="Accuracy",
    ),
    "VoxCeleb1": Data(
        prompt="Are the two speakers the same or different?",
        metric="Accuracy",
    ),
    "SpeechCommandsV1": Data(
        prompt="Classify the keyword.",
        metric="Accuracy",
    ),
    "LibriCount": Data(
        prompt="Count the amount of speakers.",
        metric="Accuracy",
    ),
    "VoxLingua33": Data(
        prompt="Classify the language.",
        metric="Accuracy",
    ),
    "SongDescriber": Data(
        prompt="Generate a caption for the music",
        metric="FENSE",
    ),
    "AISHELL-1": Data(
        prompt="Transcribe the Chinese speech into text.",
        metric="iCER",
    ),
    "ASVSpoof2015": Data(
        prompt="Detect if the audio is genuine or a spoof.",
        metric="Accuracy",
    ),
    "FSD50k": Data(
        prompt="Multi-label classification for FSD50k",
        metric="mAP",
    ),
    "FSDKaggle2018": Data(
        prompt="Multi-label classification for FSDKaggle2018",
        metric="mAP",
    ),
    "Clotho": Data(
        prompt="Generate a short caption for the audio.",
        metric="FENSE",
    ),
    "CremaD": Data(
        prompt="Identify the emotion expressed in the speech.",
        metric="Accuracy",
    ),
    "ESC-50": Data(
        prompt="Classify the environmental sound.",
        metric="Accuracy",
    ),
    "FluentSpeechCommands": Data(
        prompt="Identify the command, action, and object from the speech.",
        metric="Accuracy",
    ),
    "LibriSpeech": Data(
        prompt="Transcribe the English speech into text",
        metric="iWER",
    ),
}

DATASET_PROMPTS = {k.lower():v for k,v in DATASET_PROMPTS.items()}


def generate_path_pattern(file_paths):
    """
    Analyzes a list of full file paths and attempts to generate a single 
    POSIX-style numerical range pattern (like {START...END}). If the naming 
    convention does not support a numerical range, it returns the original 
    list of individual paths.
    
    Args:
        file_paths (list): List of full file paths (strings).
        
    Returns:
        list: A list containing a single pattern string, or the original 
              list of paths if pattern generation fails.
    """
    if not file_paths:
        return []

    # 1. Setup paths and check single file case
    filenames = sorted([Path(p).name for p in file_paths])
    base_dir = str(Path(file_paths[0]).parent)

    if len(filenames) == 1:
        return file_paths # Returns the full path of the single file

    first_name = filenames[0]
    last_name = filenames[-1]

    # 2. Find the Longest Common Prefix (LCP) and Suffix (LCS)
    prefix_len = 0
    for i in range(min(len(first_name), len(last_name))):
        if first_name[i] == last_name[i]:
            prefix_len += 1
        else:
            break
    prefix = first_name[:prefix_len]

    suffix_len = 0
    for i in range(1, min(len(first_name), len(last_name)) + 1):
        if first_name[-i] == last_name[-i]:
            suffix_len += 1
        else:
            break
    suffix = first_name[len(first_name) - suffix_len:]

    # 3. Determine the numerical range (min to max index)
    variable_parts = [name[prefix_len:len(name)-suffix_len] for name in filenames]

    parsed_numbers = []
    
    for part in variable_parts:
        if not part.isdigit():
            # Fallback: if any part is NOT numeric, we cannot form a safe numeric range.
            return file_paths 
        try:
            parsed_numbers.append(int(part))
        except ValueError:
             # Safety fallback
             return file_paths 

    
    # 4. Generate the {START...END} pattern
    if parsed_numbers:
        start_num = min(parsed_numbers)
        end_num = max(parsed_numbers)
        sequence_tag = f"{{{start_num:02d}..{end_num:02d}}}"
        pattern_filename = prefix + sequence_tag + suffix
        return [str(Path(base_dir) / pattern_filename)]
    return file_paths # Final fallback


def generate_configs(root_dir="env", output_dir="configs", workers=4):
    """
    Finds .tar.gz files in the directory structure, generates POSIX path 
    patterns, and writes the resulting YAML configurations, separating 
    training (train/valid) and evaluation (test) data.
    """
    # Create the main configs directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create the test configs subdirectory
    test_output_dir = Path(output_dir) / "test"
    test_output_dir.mkdir(exist_ok=True)
    
    organized_data = {}
    all_datasets = set()

    # Step 1: Find all files and generate patterns
    print(f"Searching for .tar.gz files in '{root_dir}'...")
    for dataset_path in Path(root_dir).iterdir():
        if dataset_path.is_dir():
            dataset_name = dataset_path.name
            
            for split_name_path in dataset_path.iterdir():
                split_name = split_name_path.name
                
                if split_name in ['train', 'test', 'valid'] and split_name_path.is_dir():
                    file_paths = [str(p) for p in split_name_path.glob("*.tar.gz")]
                    
                    if file_paths:
                        pattern_list = generate_path_pattern(file_paths)
                        
                        if split_name not in organized_data:
                            organized_data[split_name] = {}
                        
                        organized_data[split_name][dataset_name] = pattern_list
                        all_datasets.add(dataset_name)
                        
                        # Summary print for user feedback
                        summary = pattern_list[0] if len(pattern_list) == 1 else f"{len(pattern_list)} individual files"
                        print(f"  Found {len(file_paths)} files for {dataset_name}/{split_name}, condensed to pattern: {summary}")

    if not all_datasets:
        print("No datasets found to process. Exiting.")
        return

    # Step 2: Generate and save the YAML configs
    print(f"\nGenerating configurations for {len(all_datasets)} unique datasets, separating train/valid and test splits...")

    all_train_configs = {"train_data": []}

    for dataset_name in all_datasets:
        # --- A. Generate Training Config (train + valid) ---
        train_config = {}
        has_train_data = False
        for split_name in ['train']:
            if split_name in organized_data and dataset_name in organized_data[split_name]:
                has_train_data = True
                config_key = f"{split_name}_data"
                d = DATASET_PROMPTS[dataset_name.lower()]
                train_config[config_key] = {
                    dataset_name: {
                        "prompt": d.prompt,
                        "data": organized_data[split_name][dataset_name],
                        "key": d.key,
                    }
                }

                all_train_configs['train_data'].append(train_config[config_key])
        if has_train_data:
            train_config["num_training_workers"] = workers
            output_filepath = Path(output_dir) / f"{dataset_name.lower()}_config.yaml"
            with open(output_filepath, 'w') as f:
                yaml.dump(train_config, f, sort_keys=False, default_flow_style=False)
            print(f"  -> Generated Training Config: {output_filepath}")


            output_filepath_all = Path(output_dir) / f"all_train_config.yaml"
            with open(output_filepath_all, 'w') as f:
                yaml.dump(all_train_configs, f, sort_keys=False, default_flow_style=False)
            print(f"  -> Generated Training Config: {output_filepath}")


        # --- B. Generate Evaluation Config (test) ---
        if 'test' in organized_data and dataset_name in organized_data['test']:
            eval_config = {}
            # Use the requested key format: eval_datasetname (lowercased)
            eval_key = f"eval_{dataset_name.lower()}"
            # Get prompt and data from the 'test' split
            d = DATASET_PROMPTS[dataset_name.lower()]
            prompt = d.prompt
            metric =d.metric
            key = d.key
            data_list = organized_data['test'][dataset_name]

            eval_config[eval_key] = {
                "prompt": prompt,
                "key": key,
                "data": data_list,
                "batch_size": 4,
                "num_workers": 0,
                "metric": metric,
            }
            
            test_output_filepath = test_output_dir / f"{dataset_name.lower()}_test_config.yaml"

            with open(test_output_filepath, 'w') as f:
                yaml.dump(eval_config, f, sort_keys=False, default_flow_style=False)
                
            print(f"  -> Generated Evaluation Config: {test_output_filepath}")

if __name__ == "__main__":
    
    ROOT_DATA_DIR = "env"
    OUTPUT_CONFIG_DIR = "configs"

    generate_configs(ROOT_DATA_DIR, OUTPUT_CONFIG_DIR)

    print("\nYAML configuration generation complete.")
