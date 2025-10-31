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

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from .configuration_xaresllm import XaresLLMModelConfig


class XaresLLMModel(PreTrainedModel, nn.Module):
    config_class = XaresLLMModelConfig

    def __init__(self, config: XaresLLMModelConfig, audio_encoder) -> None:
        super().__init__(config)
        self.config = config
        self.audio_encoder = audio_encoder
        for param in self.audio_encoder.parameters():
            param._requires_grad = False
        self.decoder = AutoModelForCausalLM.from_pretrained(config.decoder_type)
        self.audio_projector = nn.Linear(self.audio_encoder.output_dim, self.decoder.config.hidden_size)

    def forward(self, audio, audio_attention_mask, input_ids, attention_mask, labels, **kwargs):
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                audio_feature, mel_attention_mask = self.audio_encoder(audio, audio_attention_mask)
                audio_feature = self.audio_projector(audio_feature)
        # An error occurs if .get_input_embeddings() is used with self.input_embeds = ...
        input_embeds = self.decoder.get_input_embeddings()(input_ids)  # Int -> Float

        # concatenate all data: [AUDIO, TEXT]
        input_embeds = torch.cat((audio_feature, input_embeds), dim=1)
        zero_audio_targets = torch.full(
            audio_feature.shape[:2], device=audio_feature.device, dtype=torch.int, fill_value=-100
        )
        labels = torch.cat((zero_audio_targets, labels), dim=1)
        attention_mask = torch.cat(
            (torch.ones(*audio_feature.shape[:2], device=attention_mask.device), attention_mask),
            dim=1,
        )
        print(audio_feature.shape, input_embeds.shape)

        return self.decoder(input_ids=None, inputs_embeds=input_embeds, labels=labels, attention_mask=attention_mask)

    def generate(self, mel):
        pass
