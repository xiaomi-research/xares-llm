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

from transformers.configuration_utils import PretrainedConfig
from typing import Dict, Any


class XaresLLMModelConfig(PretrainedConfig):
    model_type = "xaresllmmodel"

    def __init__(
        self,
        audio_encoder_name: str | None = None,
        audio_encoder_params: Dict[str, Any] = {},
        decoder_type: str = "gpt2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.decoder_type = decoder_type
        self.audio_encoder_name = audio_encoder_name
        self.audio_encoder_params = audio_encoder_params


__all__ = ["XAresLLMModelConfig"]
