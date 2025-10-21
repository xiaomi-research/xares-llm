import argparse
from typing import Dict, Any, List

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
import importlib
from pathlib import Path

# from xares_llm.audio_encoder_checker import check_audio_encoder
from xares_llm.task import XaresLLMTask, XaresLLMTrainConfig, XaresLLMEvaluationConfig
from xares_llm.utils import attr_from_py_path, setup_global_logger


# Mappings from config.yaml -> Path to the config. By default we store most configs in the package tree, but users can also provide their own
AVAILABLE_SINGLE_TRAINING_CONFIGS = {Path(_).stem: _ for _ in importlib.resources.files('xares_llm.tasks.single').iterdir()}
AVAILABLE_EVALUATION_CONFIGS = {Path(_).stem: _ for _ in importlib.resources.files('xares_llm.tasks.evaluation').iterdir()}


def main(args):
    setup_global_logger()
    enable_multiprocessing = args.max_jobs > 0
    torch.multiprocessing.set_start_method("spawn")
    train_config = args.train_config

    audio_encoder = attr_from_py_path(args.encoder_py, endswith="Encoder")()
    runner = XaresLLMTask(audio_encoder=audio_encoder,train_config=train_config)
    scores: List[Dict[str, Any]] = runner.run(args.eval_configs)

    logger.info("Scoring completed: All tasks scored.")

    # Print results
    df = pd.DataFrame(scores)
    df.sort_values(by="Task", inplace=True)

    df['weighted_scores'] = df['score'] * df['weight']
    new_row = pd.DataFrame({
        'Task': 'Overall',
        'score': (df['score'] * df['weight']).sum() / df['weight'].sum(),
        'weight': df['weight'].sum()})
    df = pd.concat((df, new_row), ignore_index=True)
    logger.info(f"\nResults:\n{df.to_string(index=False)}")
    df.to_csv(Path(train_config.output_dir) / 'scores.tsv', sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XARES-LLM")
    parser.add_argument(
        "encoder_py", type=str, help="Encoder path. e.g.: example/dummy/dummyencoder.py"
    )
    parser.add_argument(
        "train_config",
        type=XaresLLMTrainConfig.from_file,
        help="Tasks .yaml or predefined dataset.",
        default=AVAILABLE_SINGLE_TRAINING_CONFIGS,
    )
    parser.add_argument(
        "eval_configs",
        type=XaresLLMEvaluationConfig.configs_from_file,
        help="Evaluation Task .yaml. One Yaml can specify multiple datasets. By default we use the XARES-LLM datasets.",
        default=AVAILABLE_EVALUATION_CONFIGS,
    )
    parser.add_argument(
        "--max-jobs", type=int, default=1, help="Maximum number of concurrent tasks."
    )
    args = parser.parse_args()
    setup_global_logger()


    main(args)
