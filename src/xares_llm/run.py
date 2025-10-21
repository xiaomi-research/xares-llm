import argparse
from functools import partial
from typing import Dict, Any

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
import importlib
from pathlib import Path

# from xares_llm.audio_encoder_checker import check_audio_encoder
from xares_llm.task import XaresLLMTrainConfig, XaresLLMEvaluationConfig
from xares_llm.utils import attr_from_py_path, setup_global_logger


# Mappings from config.yaml -> Path to the config. By default we store most configs in the package tree, but users can also provide their own
AVAILABLE_SINGLE_TRAINING_CONFIGS = {Path(_).stem: _ for _ in importlib.resources.files('xares_llm.tasks.single').iterdir()}
AVAILABLE_EVALUATION_CONFIGS = {Path(_).stem: _ for _ in importlib.resources.files('xares_llm.tasks.evaluation').iterdir()}



def worker(
    encoder_py: None | str,
    task_config_path: str,
) -> Dict[str, Any]:
    # Encoder setup
    encoder = attr_from_py_path(encoder_py, endswith="Encoder")() if encoder_py else None

    # Task setup
    config = task
    task = XaresTask(config=config)





    # if do_mlp:
    #     logger.info(f"Running run_mlp for task {config.name} ...")
    #     mlp_score = task.run_mlp()
    #     logger.info(f"MLP score of {config.name}: {mlp_score}")

    # knn_score = (0, 0)
    # if do_knn and task.config.do_knn:
    #     logger.info(f"Running KNN for task {config.name} ...")
    #     knn_score = task.run_knn()
    #     logger.info(f"KNN score of {config.name}: {knn_score}")

    torch.cuda.empty_cache()
    return task.config.formal_name, mlp_score, knn_score, task.config.private


def stage_1(encoder_py, task_py, gpu_id):
    torch.cuda.set_device(gpu_id)
    return worker(encoder_py, task_py, do_encode=True)


def stage_2(encoder_py, task_py, result: dict):
    result.update({task_py: worker(encoder_py, task_py)})


def main(args):
    setup_global_logger()
    enable_multiprocessing = args.max_jobs > 0
    torch.multiprocessing.set_start_method("spawn")
    train_config = args.train_config

    for task_py in args.tasks_py:
        worker(None, task_py)

    if args.to_stage == 0:
        return

    # Double check the encoder and download the pretrained weights
    encoder = attr_from_py_path(args.encoder_py, endswith="Encoder")()
    if not check_audio_encoder(encoder):
        raise ValueError("Invalid encoder")
    del encoder

    # Stage 1: Execute make_encoded_tar
    if args.from_stage <= 1:
        try:
            if enable_multiprocessing:
                num_gpus = torch.cuda.device_count()
                with mp.Pool(processes=args.max_jobs) as pool:
                    pool.starmap(
                        stage_1,
                        [
                            (args.encoder_py, task_py, i % num_gpus)
                            for i, task_py in enumerate(args.tasks_py)
                        ],
                    )
            else:
                for task_py in args.tasks_py:
                    worker(args.encoder_py, task_py, do_encode=True)

            logger.info("Stage 1 completed: All tasks encoded.")
        except FileNotFoundError as e:
            logger.error(f"Task filename pattern error: {e}")
            return
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(
                    "CUDA out of memory. Try reducing `config.batch_size_encode` of tasks."
                )
            else:
                logger.error(f"Error in stage 1 (encode): {e} Must fix it before proceeding.")
                return
        logger.info("Stage 1 completed: All tasks encoded.")
    if args.to_stage == 1:
        return

    # Stage 2: Execute mlp and knn scoring
    if args.from_stage <= 2 and args.to_stage >= 2:
        if enable_multiprocessing:
            manager = mp.Manager()
            return_dict = manager.dict()
            with mp.Pool(processes=args.max_jobs) as pool:
                pool.starmap(
                    partial(stage_2, result=return_dict),
                    [(args.encoder_py, task_py) for task_py in args.tasks_py],
                )
        else:
            return_dict = {}
            for task_py in args.tasks_py:
                return_dict[task_py] = worker(args.encoder_py, task_py, do_mlp=True, do_knn=True)
        logger.info("Scoring completed: All tasks scored.")

        # Print results
        df = pd.DataFrame(return_dict.items(), columns=["py", "Scores"]).drop(columns=["py"])
        df["Task"] = df["Scores"].apply(lambda x: x[0])
        df["MLP_Score"] = df["Scores"].apply(lambda x: x[1][0])
        df["KNN_Score"] = df["Scores"].apply(lambda x: x[2][0])
        df["Private"] = df["Scores"].apply(lambda x: x[3] if len(x) > 3 else True)
        df.drop(columns=["Scores"], inplace=True)
        df.sort_values(by="Task", inplace=True)

        print(f"\nResults:\n{df.to_string(index=False)}")

        avg_mlp_all, avg_knn_all = weighted_average({k: v[1:-1] for k, v in return_dict.items()})
        print("\nWeighted Average MLP Score for All Datasets:", avg_mlp_all)
        print("Weighted Average KNN Score for All Datasets:", avg_knn_all)
        if any([v[-1] == True for v in return_dict.values()]):
            avg_mlp_public, avg_knn_public = weighted_average(
                {k: v[1:-1] for k, v in return_dict.items() if v[-1] == True}
            )

            print("\nWeighted Average MLP Score for Public Datasets:", avg_mlp_public)
            print("Weighted Average KNN Score for Public Datasets:", avg_knn_public)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XARES-LLM")
    parser.add_argument(
        "encoder_py", type=str, help="Encoder path. eg: example/whisper/whisper_base.py"
    )
    parser.add_argument(
        "train_config",
        type=XaresLLMTrainConfig.from_file,
        help="Tasks .yaml or predefined dataset.",
        default=AVAILABLE_SINGLE_TRAINING_CONFIGS,
    )
    parser.add_argument(
        "eval_configs",
        type=str,
        help="Evaluation Tasks .yaml. By default we use the XARES-LLM datasets.",
        default=AVAILABLE_EVALUATION_CONFIGS,
        nargs="+",
    )
    parser.add_argument(
        "--max-jobs", type=int, default=1, help="Maximum number of concurrent tasks."
    )
    parser.add_argument("--from-stage", default=0, type=int, help="First stage to run.")
    parser.add_argument("--to-stage", default=2, type=int, help="Last stage to run.")
    args = parser.parse_args()
    main(args)
