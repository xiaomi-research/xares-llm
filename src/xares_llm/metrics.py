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

from typing import Literal, List
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
import itertools
from transformers import EvalPrediction
from jiwer import cer, wer
import unicodedata
import string
import re


def preprocess_string(text: str):
    text = unicodedata.normalize("NFKC", text)  # apply NFKC
    text = text.lower()
    text = text.replace("-", " ")  # remove hyphen
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def average_precision_score_with_string(
    y_true: List[str], y_pred: List[str], num_classes: int, separator: str = ";", **kwargs
):
    unique_labels = set(itertools.chain(*[s.split(separator) for s in y_true]))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    # Encode targets and predictions into binary multi-label format
    target_tensor = np.zeros((len(y_true), num_classes), dtype=np.float32)
    pred_tensor = np.zeros((len(y_pred), num_classes), dtype=np.float32)

    for i, labels in enumerate(y_true):
        indices = [label_to_index[label] for label in labels.split(separator) if label in label_to_index]
        target_tensor[i, indices] = 1.0

    for i, labels in enumerate(y_pred):
        indices = [label_to_index[label] for label in labels.split(separator) if label in label_to_index]
        pred_tensor[i, indices] = 1.0
    return average_precision_score(target_tensor, pred_tensor, **kwargs)


class MetricRegistry:
    _registry = {}

    @classmethod
    def register(cls, class_type):
        """Decorator to register a class by name"""
        cls._registry[class_type.__name__] = class_type
        return class_type

    @classmethod
    def create(cls, name, *args, **kwargs):
        """Create an instance of a registered class"""
        if name not in cls._registry:
            raise ValueError(f"Unknown class: '{name}', possible values {list(cls._registry.keys())}")
        return cls._registry[name](*args, **kwargs)

    @classmethod
    def get_registered_names(cls):
        """Get all registered class names as a Literal[] type"""
        return Literal[tuple(cls._registry.keys())]  # Dynamic Literal type


class TokenDecoder:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def decode_predictions(self, pred: EvalPrediction) -> tuple[List[str], List[str]]:
        predictions = pred.predictions
        label_ids = pred.label_ids
        if isinstance(predictions, np.ndarray) and len(predictions.shape) > 2:
            predictions = np.argmax(predictions, axis=-1)
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        predictions[predictions == -100] = self.tokenizer.pad_token_id
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        return decoded_preds, decoded_labels


@MetricRegistry.register
class iWER:
    def __init__(self, tokenizer, **kwargs):
        self.tokendecoder = TokenDecoder(tokenizer)
        super().__init__(**kwargs)

    def __call__(self, pred: EvalPrediction):
        preds, targets = self.tokendecoder.decode_predictions(pred)
        wer_score = wer(list(map(preprocess_string, targets)), list(map(preprocess_string, preds)))
        return {"iWER": max(0, 1.0 - wer_score)}


@MetricRegistry.register
class iCER:
    def __init__(self, tokenizer, **kwargs):
        self.tokendecoder = TokenDecoder(tokenizer)
        super().__init__(**kwargs)

    def __call__(self, pred: EvalPrediction):
        preds, targets = self.tokendecoder.decode_predictions(pred)
        cer_score = cer(list(map(preprocess_string, targets)), list(map(preprocess_string, preds)))
        return {"iCER": max(0, 1.0 - cer_score)}


@MetricRegistry.register
class Accuracy:
    def __init__(self, tokenizer, **kwargs):
        self.tokendecoder = TokenDecoder(tokenizer)
        super().__init__(**kwargs)

    def __call__(self, pred: EvalPrediction):
        preds, targets = self.tokendecoder.decode_predictions(pred)
        preds = list(map(preprocess_string, preds))
        targets = list(map(preprocess_string, targets))
        return {"Accuracy": accuracy_score(targets, preds)}  # scikit supports strings


@MetricRegistry.register
class mAP:
    def __init__(self, tokenizer, num_classes: int, separator: str = ";", **kwargs):
        self.tokendecoder = TokenDecoder(tokenizer)
        self.num_classes = num_classes
        self.separator = separator
        super().__init__(**kwargs)

    def __call__(self, pred: EvalPrediction):
        preds, targets = self.tokendecoder.decode_predictions(pred)
        preds = list(map(preprocess_string, preds))
        targets = list(map(preprocess_string, targets))
        return {
            "mAP": average_precision_score_with_string(
                targets, preds, num_classes=self.num_classes, separator=self.separator
            )
        }  # scikit supports strings


@MetricRegistry.register
class FENSE:
    def __init__(self, tokenizer, **kwargs):
        self.tokendecoder = TokenDecoder(tokenizer)
        self.separator = separator
        super().__init__(**kwargs)

    def __call__(self, pred: EvalPrediction):
        preds, targets = self.tokendecoder.decode_predictions(pred)
        preds = list(map(preprocess_string, preds))
        targets = list(map(preprocess_string, targets))
        return {
            "mAP": average_precision_score_with_string(
                targets, preds, num_classes=self.num_classes, separator=self.separator
            )
        }  # scikit supports strings


RegisteredMetricsLiteral = MetricRegistry.get_registered_names()

def get_metric(name: RegisteredMetricsLiteral, **metric_kw) -> RegisteredMetricsLiteral:
    return MetricRegistry.create(name, **metric_kw)
