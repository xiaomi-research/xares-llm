import torch
from transformers import WhisperModel, WhisperProcessor

def length_to_mask(lengths: torch.Tensor, max_len: int| None = None) -> torch.Tensor:
    if max_len is None:
        max_len = lengths.amax()
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = idx < lengths.unsqueeze(1)
    return mask.long()

class WhisperEncoder(torch.nn.Module):
    def __init__(self, model_name="openai/whisper-base"):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name).get_encoder()
        self.output_dim = self.model.config.d_model

    def forward(self, audio: torch.Tensor, audio_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        # Since feature extraction is on cpu this is super slow
        assert isinstance(audio, torch.Tensor)
        audio = audio.cpu().numpy()
        if audio.ndim == 1:
            # Single audio sequence
            audio_list = [audio]
        elif audio.ndim == 2:
            # Batch of audio sequences
            audio_list = [a for a in audio]
        else:
            raise ValueError("Audio tensor must be 1D (single sequence) or 2D (batch of sequences).")
        if audio_attention_mask is None:
            audio_lens = torch.tensor([a.shape[-1] for a in audio_list])
        else:
            audio_lens = audio_attention_mask.sum(-1)
        mel_lengths = audio_lens // self.processor.feature_extractor.hop_length
        feature_lengths = (mel_lengths - 1) // 2 + 1
        feature_lengths = (feature_lengths - 1 ) //2 + 1
        trim_length = feature_lengths.amax()
        attention_mask = length_to_mask(feature_lengths)

        features = self.processor(audio_list, sampling_rate=16000, return_tensors="pt").to(self.model.device)
        output = self.model(**features).last_hidden_state
        return output[:, :trim_length,:], attention_mask

if __name__ == "__main__":
    enc = WhisperEncoder()
    q,_ = enc(torch.randn(4, 16000), length_to_mask(torch.tensor([16000, 8000,4000,2000])))
    print(q.shape, enc.output_dim)
