import torch


class DummyEncoder(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.output_dim = 256

    def forward(self, audio, audio_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        output = torch.randn(len(audio), 10, self.output_dim)
        # Do something with attention mask or just return None, both are alright
        return output, audio_attention_mask
