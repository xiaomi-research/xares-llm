# Copyright 2025 Horizon Team, MiLM Plus, Xiaomi Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from __future__ import annotations


import torch
from pathlib import Path
from transformers import AutoTokenizer, TrainingArguments
import pandas as pd
import yaml
from dataclasses import dataclass, field, asdict
from loguru import logger
from typing import Any, Dict, List

from xares_llm.utils import seed_everything, setup_global_logger
from xares_llm.audiowebdataset import AudioTextDataType, AudioTextTokenWebdataset
from xares_llm.trainer import XaresLLMTrainerEvaluator
from xares_llm.modeling_audiollm import XaresLLMModel, XaresLLMModelConfig
from xares_llm.metrics import get_metric, RegisteredMetricsLiteral, TokenDecoder
import importlib
import pprint

# Mappings from config.yaml -> Path to the config. By default we store most configs in the package tree, but users can also provide their own
AVAILABLE_TRAINING_CONFIGS = {
    "all": _ for _ in importlib.resources.files("xares_llm.tasks.all.train").iterdir()
} | {
    str(Path(_).stem).replace("_config", ""): _
    for _ in importlib.resources.files("xares_llm.tasks.single.train").iterdir()
}
AVAILABLE_EVALUATION_CONFIGS = {"all": _ for _ in importlib.resources.files("xares_llm.tasks.all.eval").iterdir()} | {
    str(Path(_).stem).replace("_test_config", ""): _
    for _ in importlib.resources.files("xares_llm.tasks.single.eval").iterdir()
}


@dataclass
class XaresLLMTrainConfig:
    audio_encoder_module_path: str  # path to the audio encoder
    audio_encoder_kwargs: Dict[str,Any] = field(default_factory=lambda: dict())
    output_dir: str = "experiments/"
    config_name: str = "default"  # Will be set if loaded from a .yaml

    # General
    torch_num_threads: int = 1  # Do not use too many otherwise slows down
    seed: int = 42  # manual seed for all experiments

    train_data: List[AudioTextDataType] | None = None

    # decoder
    decoder_model_name: str = "Qwen/Qwen3-0.6B"

    # Dataloader/dataset arguments
    seed: int = field(default=42)
    crop_audio_length: float = 30  # Cropping all audio to at most 30s
    save_total_limit: int | None = field(default=4)
    save_steps: float = field(default=200)  # TrainingArguments is float ....
    warmup_steps: int = field(default=200)
    max_steps: int = field(
        default=10000,
        metadata={"help": "Total number of training steps to perform (default: 10k)."},
    )
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per device during training (default: 4)."}
    )

    # Optimizer
    optimizer: str = "adamw_torch"  # adamw_bnb_8bit
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    seed: int = field(default=42)
    torch_compile: bool = field(default=False)
    bf16: bool = False  # Will be set automatically
    fp16: bool = False  # Will be set automatically
    max_grad_norm: float = field(default=1.0)
    logging_dir: str = "log"
    logging_steps: int = 100
    num_training_workers: int = 0
    sort_by_length: int = 128  # Sort 128 samples by length

    def __post_init__(self):
        # torch.cuda.is_bf16_supported() does return True on V100, support is there ... but no speedup
        if torch.cuda.is_available():
            has_bf16support = torch.cuda.get_device_capability(torch.device("cuda"))[0] > 7
            if has_bf16support:
                self.bf16 = True
                self.fp16 = False
            else:
                self.fp16 = True
                self.bf16 = False

        if isinstance(self.train_data, dict):
            self.train_data = [AudioTextDataType(name=k, **val) for k, val in self.train_data.items()]
        torch.set_num_threads(self.torch_num_threads)

        setup_global_logger()
        seed_everything(self.seed)
        logger.info(f"Mixed precision training: BF16={self.bf16} FP16={self.fp16}")

    def __repr__(self):
        return pprint.pformat(asdict(self))

    @classmethod
    def from_file(cls, config_file: str, encoder_path:str, model_kwargs:Dict[str, Any] | None = None, overwrite_kwargs:Dict[str, Any] | None = None) -> XaresLLMTrainConfig:
        with open(config_file) as con_read:
            yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
        yaml_config["config_name"] = Path(config_file).stem
        yaml_config['audio_encoder_module_path'] = encoder_path
        if model_kwargs is None:
            model_kwargs = dict()
        yaml_config['audio_encoder_kwargs'] = model_kwargs
        if overwrite_kwargs is None:
            overwrite_kwargs = dict()
        yaml_config = dict(**yaml_config, **overwrite_kwargs)
        return cls(**yaml_config)

    @classmethod
    def from_file_or_key(cls, config_identifier: str, encoder_path:str, model_kwargs:Dict[str,Any] | None = None, overwrite_kwargs:Dict[str, Any] | None = None) -> XaresLLMTrainConfig:
        if config_identifier in AVAILABLE_TRAINING_CONFIGS:
            return cls.from_file(AVAILABLE_TRAINING_CONFIGS[config_identifier], encoder_path=encoder_path, model_kwargs=model_kwargs, overwrite_kwargs=overwrite_kwargs)
        path_obj = Path(config_identifier)
        if path_obj.is_file():
            return cls.from_file(config_identifier, encoder_path=encoder_path, model_kwargs=model_kwargs, overwrite_kwargs=overwrite_kwargs)
        raise ValueError(f"Unknown config identifier {config_identifier}")


@dataclass
class XaresLLMEvaluationConfig:
    data: AudioTextDataType
    metric: RegisteredMetricsLiteral
    metric_args: Dict[str, Any] = field(default_factory=lambda: dict())
    batch_size: int = 32
    num_workers: int = 0
    weight: float = 1

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
            evaluation_configs.append(cls(data=AudioTextDataType(name=k, **data_kwargs), metric=metric, **values))
        return evaluation_configs

    def __repr__(self):
        return pprint.pformat(asdict(self))

    @classmethod
    def configs_from_file_or_key(cls, config_identifier: str) -> List[XaresLLMEvaluationConfig]:
        if config_identifier in AVAILABLE_EVALUATION_CONFIGS:
            return cls.configs_from_file(AVAILABLE_EVALUATION_CONFIGS[config_identifier])
        path_obj = Path(config_identifier)
        if path_obj.is_file():
            return cls.configs_from_file(config_identifier)
        raise ValueError(f"Unknown config identifier {config_identifier}")


class XaresLLMTask:
    def __init__(self, train_config: XaresLLMTrainConfig):
        self.train_config = train_config
        if Path(self.train_config.audio_encoder_module_path).is_file():
            model_name = str(Path(self.train_config.audio_encoder_module_path).stem)
        else:
            model_name = self.train_config.audio_encoder_module_path.split('.')[-1]
        self.output_dir = Path(train_config.output_dir) / train_config.config_name / model_name 
        logger.add(
                self.output_dir / "log.txt",
                enqueue=True,
                level="INFO",
                format="[{level} {time:YYYY-MM-DD HH:mm:ss}] {message}",
            )
        logger.info(f"Experiment output path set to {self.output_dir}")
        logger.info(f"Loading {train_config.decoder_model_name} tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(train_config.decoder_model_name)
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=self.train_config.learning_rate, 
            per_device_train_batch_size=self.train_config.per_device_train_batch_size,
            save_total_limit=self.train_config.save_total_limit,
            save_steps=self.train_config.save_steps,
            lr_scheduler_type='cosine',
            warmup_steps=self.train_config.warmup_steps,
            max_grad_norm=self.train_config.max_grad_norm,
            max_steps=self.train_config.max_steps,
            optim=self.train_config.optimizer,
            weight_decay=self.train_config.weight_decay,
            seed=self.train_config.seed,
            logging_steps= self.train_config.logging_steps,
            torch_compile=self.train_config.torch_compile,
            bf16=self.train_config.bf16,
            fp16=self.train_config.fp16,
            logging_dir=Path(self.output_dir) / self.train_config.logging_dir,
        )
        # Lazy init, during .train() or .eval()
        model_init_function = lambda : XaresLLMModel(
                config=XaresLLMModelConfig(decoder_type=self.train_config.decoder_model_name, audio_encoder_name=self.train_config.audio_encoder_module_path, audio_encoder_params=self.train_config.audio_encoder_kwargs),
            )
        self.trainer = XaresLLMTrainerEvaluator(model=None, model_init=model_init_function, args=training_args)

    def run_mlp(self, eval_configs: List[XaresLLMEvaluationConfig]) -> List[Dict[str, Any]]:
        if not isinstance(eval_configs, list):
            eval_configs = [eval_configs]

        result = []
        model = self.train_mlp()
        for eval_config in eval_configs:
            dataset_name = eval_config.data.name
            score, output_df = self.evaluate_mlp(trained_model=model, eval_config=eval_config)
            logger.info(f"{dataset_name}: [{eval_config.metric}]: {score:.2f}")
            result.append({"Task": dataset_name, "score": score, "weight": eval_config.weight})
            output_df.to_csv(self.output_dir / f'predictions_{dataset_name}.csv', index=False)
            logger.debug(f"Model outputs can be seen in {self.output_dir / f'predictions_{dataset_name}.csv'}" )
        return result

    def train_mlp(self) -> XaresLLMModel:
        train_data_object = AudioTextTokenWebdataset(
            data_urls=self.train_config.train_data,
            tokenizer=self.tokenizer,
            training=True,
            batch_size=self.train_config.per_device_train_batch_size,
            resample=True,
            sort_by_length=self.train_config.sort_by_length,
            num_workers=self.train_config.num_training_workers,
            crop_audio_length=self.train_config.crop_audio_length,
        )
        self.trainer.train_data_object = train_data_object
        self.trainer.train()
        logger.info(f"Finished training: {self.output_dir}")
        return self.trainer.model

    def evaluate_mlp(
        self,
        eval_config: XaresLLMEvaluationConfig,
        trained_model: XaresLLMModel | None = None,
        chpt_path: str | Path | None = None,
    ) -> tuple[Dict[RegisteredMetricsLiteral, float], pd.DataFrame]:
        if trained_model is not None:
            model = trained_model
        elif chpt_path is not None:
            logger.info(f"Loaded model parameters from {chpt_path}")
            model = XaresLLMModel.from_pretrained(chpt_path)
        else:
            model = self.trainer.model

        self.trainer.model = model

        metrics_compute_function = get_metric(eval_config.metric, tokenizer=self.tokenizer, **eval_config.metric_args)

        data_object_eval = AudioTextTokenWebdataset(
            data_urls=eval_config.data,
            tokenizer=self.tokenizer,
            training=False,
            batch_size=eval_config.batch_size,
            sort_by_length=256, # just to speed up a bit
            num_workers=eval_config.num_workers,
        )
        # remove LoRA to speed up inference
        self.trainer.compute_metrics = metrics_compute_function

        result = self.trainer.predict(test_dataset = data_object_eval)

        decoder = TokenDecoder(self.tokenizer)
        prediced_text, targets = decoder.decode_predictions(result)

        prediction_df = pd.DataFrame({'predict':prediced_text, 'labels':targets})
        return result.metrics[f'test_{eval_config.metric}'], prediction_df

    def run(self, eval_configs: List[XaresLLMEvaluationConfig]):
        scores = self.run_mlp(eval_configs=eval_configs)
        return scores
