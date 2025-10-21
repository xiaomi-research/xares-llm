from transformers.configuration_utils import PretrainedConfig


class XaresLLMModelConfig(PretrainedConfig):
    model_type = "xaresllmmodel"

    def __init__(
        self,
        decoder_type: str = "gpt2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.decoder_type = decoder_type


__all__ = ["XAresLLMModelConfig"]
