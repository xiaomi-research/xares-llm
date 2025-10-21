from __future__ import annotations

import torch
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, TrainingArguments
import yaml
from dataclasses import dataclass, field
from loguru import logger
from typing import Any, Callable, Dict, List

from xares_llm.audiowebdataset import AudioTextDataType, AudioTextTokenWebdataset
from xares_llm.utils import seed_everything, setup_global_logger
from xares_llm.trainer import XaresLLMEvaluator, XaresLLMTrainer
from xares_llm.modeling_audiollm import XaresLLMModel, XaresLLMModelConfig
from xares_llm.metrics import get_metric, RegisteredMetricsLiteral


@dataclass
class XaresLLMTrainConfig:
    output_dir: str = "experiments/"
    env_root: Path | str = Path("env/")
    config_name: str | None = None  # Will be set if loaded from a .yaml

    # General
    private: bool = False
    torch_num_threads: int = 2  # Do not use too many otherwise slows down
    seed: int = 42  # manual seed for all experiments
    eval_weight: int = 0

    train_data: List[AudioTextDataType] | None = None
    valid_data: List[AudioTextDataType] | None = None

    # Audio tar
    force_download: bool = False
    zenodo_id: str | None = None

    # MLP
    decoder_model_name: str = "gpt2"
    ckpt_dir_name = "checkpoints"
    embedding_dir_name = "embeddings"
    ckpt_name = "best.ckpt"

    # Dataloader/dataset arguments
    seed: int = field(default=42)
    save_total_limit: int | None = field(default=4)
    save_steps: float = field(default=1000)  # TrainingArguments is float ....
    warmup_steps: int = field(default=1000)
    max_steps: int = field(
        default=200000,
        metadata={"help": "Total number of training steps to perform (default: 200k)."},
    )
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per device during training (default: 16)."}
    )

    # Optimizer
    optimizer: str = "adamw_torch"  # adamw_bnb_8bit
    learining_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    seed: int = field(default=42)
    torch_compile: bool = field(default=False)
    bf16: bool = False
    fp16: bool = False
    max_grad_norm: float = field(default=1.0)
    logging_dir: str = "log"
    gradient_accumulation_steps: int = field(default=1)

    batch_size_train: int = 16
    batch_size_valid: int | None = None
    learning_rate: float = 1e-4
    iterations: int = 100_000
    valid_every: int = 1000
    num_training_workers: int = 0
    num_validation_workers: int = 0
    sort_by_length: bool = True
    # metric: METRICS_TYPE = "accuracy"
    metric_args: Dict[str, Any] = field(default_factory=lambda: dict())

    def __post_init__(self):
        if isinstance(self.train_data, dict):
            self.train_data = [
                AudioTextDataType(name=k, **val) for k, val in self.train_data.items()
            ]
        if isinstance(self.valid_data, dict):
            self.valid_data = [
                AudioTextDataType(name=k, **val) for k, val in self.valid_data.items()
            ]
        # if isinstance(self.test_data, dict):
        #     self.test_data = [AudioTextDataType(name=k, **val) for k, val in self.test_data.items()]

        setup_global_logger()
        if self.batch_size_valid is None:
            self.batch_size_valid = self.batch_size_train
        if self.env_root is None:
            self.env_root = Path("env")
        torch.set_num_threads(self.torch_num_threads)
        seed_everything(self.seed)

    @classmethod
    def from_file(cls, config_file: str) -> XaresLLMTrainConfig:
        with open(config_file) as con_read:
            yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
        yaml_config["config_name"] = Path(config_file).stem
        return cls(**yaml_config)


@dataclass
class XaresLLMEvaluationConfig:
    data: AudioTextDataType
    metric: RegisteredMetricsLiteral
    batch_size: int = 1
    num_workers: int = 0

    @classmethod
    def configs_from_file(cls, yaml_config_file: str) -> List[XaresLLMEvaluationConfig]:
        with open(yaml_config_file) as con_read:
            yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
        evaluation_configs = []
        # Yaml config should have data-name as key (config_name)
        for k, values in yaml_config.items():
            try:
                data_kwargs = values.pop("data")
                metric = values.pop("metric")
            except KeyError as e:
                logger.exception(
                    f"data and metric are required keys! Check config {yaml_config_file}\nMissing key: {e}"
                )
                raise KeyError(e)
            evaluation_configs.append(
                cls(data=AudioTextDataType(name=k, **data_kwargs), metric=metric, **values)
            )
        return evaluation_configs


class XaresLLMTask:
    def __init__(self, audio_encoder: Callable, train_config: XaresLLMTrainConfig):
        self.audio_encoder = audio_encoder
        self.train_config = train_config
        self.tokenizer = AutoTokenizer.from_pretrained(train_config.decoder_model_name)

    def download_audio_tar(self):
        pass
        # audio_ready_path = self.config.env_dir / self.config.xares_settings.audio_ready_filename
        # if not self.config.force_download and audio_ready_path.exists():
        #     logger.warning(f"Skip downloading audio tar: {audio_ready_path} exists.")
        #     return
        #
        # download_zenodo_record(self.config.zenodo_id, self.env_dir, force_download=self.config.force_download)

    def run_mlp(self, eval_configs: List[XaresLLMEvaluationConfig]) -> List[Dict[str, Any]]:
        if not isinstance(eval_configs, list):
            eval_configs = [eval_configs]

        result = []
        model = self.train_mlp()
        for eval_config in eval_configs:
            dataset_name = eval_config.data.name
            score = self.evaluate_mlp(trained_model=model, eval_config=eval_config)

            logger.info(f"{dataset_name}: [{eval_config.metric}]: {score:.2f}")
            result.append({"dataset": dataset_name, "score": score})
        return result

    def train_mlp(self) -> XaresLLMModel:
        model = XaresLLMModel(
            config=XaresLLMModelConfig(decoder_type=self.train_config.decoder_model_name),
            audio_encoder=self.audio_encoder,
        )
        tokenizer = AutoTokenizer.from_pretrained(model.config.decoder_type)
        data_object = AudioTextTokenWebdataset(
            data_urls=self.train_config.train_data,
            tokenizer=tokenizer,
            training=True,
            batch_size=self.train_config.batch_size_train,
            resample=True,
            sort_by_length=256,
            num_workers=self.train_config.num_training_workers,
        )

        training_args = TrainingArguments(
            output_dir=str(self.train_config.output_dir),
            learning_rate=self.train_config.learining_rate,  # Using the typo'd attribute value
            per_device_train_batch_size=self.train_config.per_device_train_batch_size,
            # Remaining arguments
            save_total_limit=self.train_config.save_total_limit,
            save_steps=self.train_config.save_steps,
            warmup_steps=self.train_config.warmup_steps,
            max_steps=self.train_config.max_steps,
            optim=self.train_config.optimizer,
            weight_decay=self.train_config.weight_decay,
            seed=self.train_config.seed,
            save_safetensors=False,
            torch_compile=self.train_config.torch_compile,
            bf16=self.train_config.bf16,
            logging_dir=Path(self.train_config.output_dir) / self.train_config.logging_dir,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
        )

        trainer = XaresLLMTrainer(model, training_args, train_data_object=data_object)
        trainer.train()
        return model

    def evaluate_mlp(
        self,
        eval_config: XaresLLMEvaluationConfig,
        trained_model: XaresLLMModel | None = None,
        chpt_path: str | Path | None = None,
    ) -> Dict[RegisteredMetricsLiteral, float]:
        if trained_model is not None:
            model = trained_model
        elif chpt_path is not None:
            logger.info(f"Loaded model parameters from {chpt_path}")
            model = XaresLLMModel.from_pretrained(chpt_path, audio_encoder=self.audio_encoder)
        else:
            raise ValueError("You need to provide either trained_model or chpt_path.")

        tokenizer = AutoTokenizer.from_pretrained(model.config.decoder_type)
        metrics_compute_function = get_metric(eval_config.metric, tokenizer=tokenizer)

        data_object_eval = AudioTextTokenWebdataset(
            data_urls=eval_config.test_data,
            tokenizer=tokenizer,
            training=False,
            batch_size=eval_config.batch_size,
            sort_by_length=256,
            num_workers=eval_config.num_workers,
        )
        evaluator = XaresLLMEvaluator(
            model, data_object_eval=data_object_eval, compute_metrics=metrics_compute_function
        )
        return evaluator.evaluate()

    def run(self, eval_configs: List[XaresLLMEvaluationConfig]):
        # self.download_audio_tar()
        scores = self.run_mlp(eval_configs=eval_configs)
        return scores
