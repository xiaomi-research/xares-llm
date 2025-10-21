from typing import Literal, List
import numpy as np
from transformers import EvalPrediction
from jiwer import cer, wer
import unicodedata
import re


def preprocess_string(text: str):
    text = unicodedata.normalize("NFKC", text)  # apply NFKC
    text = text.lower()
    text = text.replace("-", " ")  # remove hyphen
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
            raise ValueError(
                f"Unknown class: '{name}', possible values {list(cls._registry.keys())}"
            )
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
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        return decoded_preds, decoded_labels


@MetricRegistry.register
class WER:
    def __init__(self, tokenizer, **kwargs):
        self.tokendecoder = TokenDecoder(tokenizer)
        super().__init__(**kwargs)

    def __call__(self, pred: EvalPrediction):
        preds, targets = self.tokendecoder.decode_predictions(pred)
        return {
            "WER": wer(list(map(preprocess_string, targets)), list(map(preprocess_string, preds)))
        }


@MetricRegistry.register
class CER:
    def __init__(self, tokenizer, **kwargs):
        self.tokendecoder = TokenDecoder(tokenizer)
        super().__init__(**kwargs)

    def __call__(self, pred: EvalPrediction):
        preds, targets = self.tokendecoder.decode_predictions(pred)
        return {
            "CER": cer(list(map(preprocess_string, targets)), list(map(preprocess_string, preds)))
        }


@MetricRegistry.register
class Accuracy:
    def __init__(self, tokenizer, **kwargs):
        self.tokendecoder = TokenDecoder(tokenizer)
        super().__init__(**kwargs)

    def __call__(self, pred: EvalPrediction):
        preds, targets = self.tokendecoder.decode_predictions(pred)
        return {
            "CER": cer(list(map(preprocess_string, targets)), list(map(preprocess_string, preds)))
        }


RegisteredMetricsLiteral = MetricRegistry.get_registered_names()


def get_metric(name: RegisteredMetricsLiteral, **metric_kw) -> RegisteredMetricsLiteral:
    return MetricRegistry.create(name, **metric_kw)
