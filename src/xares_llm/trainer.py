from transformers import AutoTokenizer, TrainingArguments, Trainer
from xares_llm.audiowebdataset import AudioTextTokenWebdataset


class XaresLLMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.train_data_object: AudioTextTokenWebdataset = kwargs.pop("train_data_object")
        super().__init__(train_dataset=self.train_data_object.create_dataset(), *args, **kwargs)

    def get_train_dataloader(self):
        return self.train_data_object.create_dataloader()

    def evaluate(self, *args, **kwargs):
        raise ValueError("Only .train() is supported.")


class XaresLLMEvaluator(Trainer):
    def __init__(self, *args, **kwargs):
        self.data_object_eval = kwargs.pop("data_object_eval", None)
        super().__init__(*args, **kwargs)

    def train(self, *args, **kwargs):
        raise ValueError("Only .evaluate() is supported.")

    def get_eval_dataloader(self, *args, **kwargs):
        return self.data_object_eval.create_dataloader()
