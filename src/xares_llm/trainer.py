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

from loguru import logger
from transformers import Trainer, ProgressCallback
from xares_llm.audiowebdataset import AudioTextTokenWebdataset

class LoguruMetricsCallback(ProgressCallback):

    def on_log(self, args, state, control, logs = None, **kwargs):
        if state.is_world_process_zero:
            shallow_logs = {}
            for k, v in logs.items():
                if isinstance(v, float):
                    shallow_logs[k] = f"{v:.4g}"
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            log = ", ".join([f"{key} = {value}" for key, value in shallow_logs.items()])
            logger.info(str(log))

class XaresLLMTrainerEvaluator(Trainer):
    def __init__(self, *args, **kwargs):
        self.train_data_object: AudioTextTokenWebdataset = kwargs.pop("train_data_object", None)
        self.eval_data_object: AudioTextTokenWebdataset = kwargs.pop("eval_data_object", None)
        train_dataset = self.train_data_object.create_dataset() if self.train_data_object else None
        super().__init__(train_dataset=train_dataset, *args, **kwargs)
        self.remove_callback(ProgressCallback)
        self.add_callback(LoguruMetricsCallback)

    def get_train_dataloader(self):
        return self.train_data_object.create_dataloader()

    def get_eval_dataloader(self, eval_dataset: AudioTextTokenWebdataset, *args, **kwargs):
        return eval_dataset.create_dataloader()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        generated_ids = model.generate(**inputs, repetition_penalty=1.05, max_length=256)
        labels = inputs.get('labels')
        return (None, generated_ids, labels)
