import argparse
import sys
from typing import Literal
from tqdm import tqdm
import webdataset as wds
from xares_llm.audiowebdataset import download_data_to_cache, expand_path, SequentialDatasetSampler, CACHE_DIR
from xares_llm.task import XaresLLMTrainConfig, XaresLLMEvaluationConfig


def download(config_name: Literal['all','task1','task2'], num_workers: int, cache_dir: str):
    """
    Downloads the datasets specified in the training configuration file
    and iterates through them using a WebLoader to trigger caching.

    Args:
        config_path (str): Path to the XaresLLMTrainConfig YAML file.
        num_workers (int): Number of workers to use for the WebLoader.
        cache_dir (str): Directory where the downloaded data should be cached.
    """
    print(f"--- Configuration Summary ---")
    print(f"Cache Directory: {cache_dir}")
    print(f"WebLoader Workers: {num_workers}")
    print("-----------------------------\n")

    trainconfig = XaresLLMTrainConfig.from_file_or_key(
        config_name, 
        encoder_path="",
    )
    evalconfigs = XaresLLMEvaluationConfig.configs_from_file_or_key(config_identifier=config_name)

    # 1. Prepare Datasets and Trigger download_data_to_cache
    datasets = []
    print("\nStarting dataset preparation (downloading to cache)...")
    for data_type in trainconfig.train_data:
        ds = download_data_to_cache(expand_path(data_type), cache_dir=cache_dir)
        datasets.append((data_type, ds))
    for evalconf in evalconfigs:
        eval_data_type = evalconf.data
        eval_ds = download_data_to_cache(expand_path(eval_data_type), cache_dir=cache_dir)
        datasets.append((eval_data_type, eval_ds))


    if not datasets:
        print("\nNo valid datasets were found or processed. Exiting.")
        return

    # 2. Create Sequential Sampler and WebLoader
    dataset_sampler = SequentialDatasetSampler(datasets=datasets)
    data_loader = wds.WebLoader(
        dataset_sampler, 
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=8,
    )

    # 3. Iterate through the DataLoader to trigger actual loading/caching
    print("\nStarting full iteration through DataLoader to ensure complete caching...")
    try:
        # Iterate over the loader using tqdm for progress tracking
        for _ in tqdm(data_loader, desc="Downloading"):
            pass
        print("\n✅ **Success!** All datasets have been iterated and should be cached.")
    except Exception as e:
        print(f"\n❌ **Error during DataLoader iteration/caching**: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Download and cache webdatasets specified in a XaresLLM training config."
    )
    parser.add_argument(
        "--data", 
        type=str,
        default='all',
        choices=['all','task1','task2'],
        help=f"Path to the XaresLLMTrainConfig YAML file. (Default: all)"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=16, 
        help=f"Number of worker processes for the WebLoader. (Default: 16)"
    )

    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=CACHE_DIR, 
        help=f"Local directory to cache the downloaded data. (Default: {CACHE_DIR})"
    )

    args = parser.parse_args()
    download(
        config_name=args.data, 
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )

if __name__ == "__main__":
    main()

