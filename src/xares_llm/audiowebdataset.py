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


from typing import Any, Callable, Dict, Iterable, List, Tuple
import webdataset as wds
from webdataset.tariterators import tar_file_expander, group_by_keys
from webdataset.filters import pipelinefilter
from urllib.parse import urlparse

from functools import partial
import tempfile
import os
import random
import re
import torch
from loguru import logger
import numpy as np
import torchaudio
from torch import Tensor
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer


def set_cache() -> str:
    xdg_cache_home = os.getenv("XARES_DATA_HOME")
    if xdg_cache_home:
        cache_dir = Path(xdg_cache_home)
    else:
        cache_dir = Path.cwd() / "xares_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


CACHE_DIR = set_cache()


@dataclass
class AudioTextDataType:
    data: List[str | Path]
    prompt: str | List[str] | List[Dict[str, str]] = ""
    key: str | None = None  # Can be either a single field (labels) or a selection: (labels;label)
    name: str | None = None  # Just a filler to name the data
    prob: float | None = None
    repo_id: str | None = None  # Repo id in case of a hf repo


InputUrlType = List[AudioTextDataType] | AudioTextDataType | List[str] | str


def parse_input_to_datatype(data_urls: InputUrlType) -> List[AudioTextDataType]:
    if isinstance(data_urls, str):
        return AudioTextDataType(data=[data_urls])
    elif isinstance(data_urls, AudioTextDataType):
        return [data_urls]
    elif isinstance(data_urls, list) and all(isinstance(item, str) for item in data_urls):
        return [AudioTextDataType(data=[url]) for url in data_urls]
    else:
        return data_urls


def resolve_url(file_path: str, repo_id: str | None = None, repo_type_url_segment: str = "datasets") -> str:
    if repo_id is None:
        return file_path
    from urllib.parse import urljoin

    # So hf_hub_url will malform {0..10}.tar, thus we use a custom function
    # from huggingface_hub import hf_hub_url
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    path_to_file = f"/{repo_type_url_segment}/{repo_id}/resolve/main/{file_path}"
    base_url = urljoin(hf_endpoint, path_to_file)
    download_url = f"{base_url}?download=true"
    return download_url


def _is_corrupted_text(text: str) -> bool:
    """
    Simple corruption detector that looks for:
    1. Too many combined diacritical marks (e.g., A̳ͦ)
    2. Excessive non-alphanumeric characters
    3. Strange repeating patterns
    """
    if not text:
        return False
    # Count weird Unicode combining characters
    weird_chars = sum(1 for c in text if ord(c) > 0x0300 and ord(c) < 0x0370)

    # Count normal letters/numbers
    normal_chars = sum(1 for c in text if c.isalnum())
    # Usually only things like:[- I am., oh [__], of a [__] up, [__] sake, holy [__]]
    # trigger the following
    if weird_chars > 3 or (len(text) - normal_chars) / len(text) > 0.5:
        return True

    # Check for repeating nonsense patterns (like "AͦAͦAͦ")
    if len(text) > 10 and len(set(text)) < 4:
        return True
    return False


def exists(x: Any) -> bool:
    return x is not None


def _random_crop_audio(audio: np.ndarray, max_audio_length: float, sr: int):
    max_audio_length_samples = int(max_audio_length * sr)
    if audio.shape[-1] > max_audio_length_samples:
        start = random.randint(0, audio.shape[-1] - max_audio_length_samples)
        audio = audio[start : start + max_audio_length_samples]
    return audio


# Returns a key and maps to a List[str]
def _retrieve_key_or_not(adict: Dict, somekey: str) -> List[str] | None:
    value = adict.get(somekey, None)
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        return value
    return None


def preprocess_audio(
    audio_sr: Tuple[Tensor, int],
    mono: bool = True,
    normalize_clipped: bool = False,
    crop_audio_length: float | None = None,
    target_sample_rate: int = 16000,
) -> Tuple[Tensor, int]:
    audio, sr = audio_sr
    if mono and audio.ndim == 2:
        audio = audio.mean(0)
    if audio.abs().max() > 1.0 and normalize_clipped:
        audio = audio / audio.abs().max()
    if exists(crop_audio_length):
        audio = _random_crop_audio(audio, max_audio_length=crop_audio_length, sr=sr)
    return torchaudio.functional.resample(audio, sr, target_sample_rate), target_sample_rate


# Same as wds, but added useless extension check for opus
def decode_torch_audio(key: str, data: bytes):
    extension = re.sub(r".*[.]", "", key)
    if extension not in ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma", "opus"]:
        return None
    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
        return torchaudio.load(fname)


def text_from_json(
    sample_at_it,
    data_key: str | None = None,
    default_key_names: tuple[str, ...] = ("captions", "caption", "text"),
) -> List[str]:
    custom_keys = data_key.split(";") if data_key else []  # Split with ;
    all_keys_to_check = custom_keys + list(default_key_names)

    for key in all_keys_to_check:
        retrieved_sample = _retrieve_key_or_not(sample_at_it, key)
        if retrieved_sample:
            return retrieved_sample
    raise ValueError(
        f"Couldn't parse labels for key {data_key} [default: {default_key_names}] with input {list(sample_at_it.keys())}"
    )


def filter_audio(
    audio: Tuple[Tensor, int],
    min_audio_length: float | None = None,
    drop_clipped: bool | None = None,
) -> str | None:
    audio_sample, sr = audio
    audio_length = audio_sample.shape[-1] / sr
    if exists(min_audio_length) and audio_length < min_audio_length:
        return f"Audio too short ({audio_length} < {min_audio_length})"
    if audio_sample.abs().max() > 1.0 and drop_clipped:
        return f"Dropping clipped {audio_sample.abs().max()}"
    if audio_sample.numel() == 0:
        return f"Audio has length 0"
    return None


def filter_text_corrupt_length(
    text: str,
    max_text_length: int | None = None,
) -> None | str:
    if _is_corrupted_text(text):
        return "Text is corrupt"
    if exists(max_text_length):
        text_length = len(text)
        if text_length > max_text_length:
            return f"Text too long ({text_length} > {max_text_length})"
    return None


# Utils
def warn_and_continue(exn):
    if isinstance(exn, wds.autodecode.DecodingError):
        logger.warning(f"Warning, trouble decoding Tar: {exn.url} File: {exn.key} Key: {exn.k}")
    else:
        logger.warning(f"{repr(exn)}")

    return True


def length_to_mask(length: torch.Tensor, max_length: int) -> torch.Tensor:
    mask = (torch.arange(max_length, device=length.device).expand(len(length), max_length) < length.unsqueeze(1)).int()
    return mask


# Function roughly est the input length to the model.
def _sort_by_audio_length(item, key: str = "audio", dim: int = -1):
    return max(1, item[key].shape[dim])


def _sort_by_length(
    data,
    bufsize=128,
    reverse: bool = True,
    sort_function: Callable = _sort_by_audio_length,
):
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= bufsize:
            buf.sort(key=lambda item: sort_function(item), reverse=reverse)
            yield from buf
            buf = []
    if buf:
        buf.sort(key=lambda item: sort_function(item), reverse=reverse)
        yield from buf
        buf = []


sort_by_length = wds.pipelinefilter(_sort_by_length)


def _pad(tensorlist: List[torch.Tensor | List], padding_value: float = 0.0, max_length: int | None = None):
    # Tensors are expected to be B, ..., T
    if not isinstance(tensorlist[0], torch.Tensor):
        tensorlist = [torch.tensor(seq) for seq in tensorlist]
    lengths = [f.shape[-1] for f in tensorlist]
    dims = tensorlist[0].shape
    trailing_dims = dims[:-1]
    batch_dim = len(lengths)
    if max_length is None:
        to_pad_to_length = max(lengths)
    else:
        to_pad_to_length = max_length
    out_dims = (batch_dim,) + trailing_dims + (to_pad_to_length,)
    out_tensor = torch.full(out_dims, fill_value=padding_value, dtype=tensorlist[0].dtype)
    for i, tensor in enumerate(tensorlist):
        length = tensor.shape[-1]
        out_tensor[i, ..., :length] = tensor[..., :length]
    return out_tensor, length_to_mask(torch.as_tensor(lengths), max_length=to_pad_to_length)


def _process_sample_stream(
    stream: Iterable[Dict[str, Any]],
    tokenizer: Callable,
    min_audio_length: float | None = None,
    drop_clipped: bool = True,
    normalize_clipped: bool = True,
    crop_audio_length: float | None = None,
    max_text_token_length: int | None = None,
    text_data_key: str | None = None,
    target_sample_rate: int = 16000,
    mono: bool = True,
    prompt: str | List[str] | Dict[str, Any] = "",
    append_targets_to_input: bool = True,  # During training yes, inference no
    handler: Callable = warn_and_continue,
    tokenizer_eos_token: bool = False,
) -> Iterable[Dict[str, Any]]:
    for data_sample in stream:
        # Note: We use the local variables from the outer scope directly.
        # 1. Pop and Extract Data
        audio = data_sample.pop("audio")
        tarname = data_sample.pop("tar")
        filename = data_sample.pop("filename")
        texts = text_from_json(data_sample.pop("text"), data_key=text_data_key)
        # Texts = List[str], Can be multiple for Captioning data
        if audio_filter_reason := filter_audio(audio, min_audio_length=min_audio_length, drop_clipped=drop_clipped):
            logger.warning(
                f"Dropped sample {data_sample.get('filename', 'unknown')} in {tarname} Reason: {audio_filter_reason}"
            )
            continue
        # Audio reprocessing
        audio, _ = preprocess_audio(
            audio,
            mono=mono,
            normalize_clipped=normalize_clipped,
            crop_audio_length=crop_audio_length,
            target_sample_rate=target_sample_rate,
        )
        for text in texts:
            # Filtering Text, Here only for corrupt samples, since length is for tokens
            if text_filter_reason := filter_text_corrupt_length(
                text,
            ):
                logger.warning(
                    f"Dropped sample {data_sample.get('filename', 'unknown')} in {tarname} Reason: {text_filter_reason}"
                )
                continue

            sample_prompt = prompt
            if prompt == "" and "prompt" in data_sample:
                sample_prompt = data_sample.pop("prompt")
            if isinstance(sample_prompt, list) and len(sample_prompt) > 0 and isinstance(sample_prompt[0], str):
                sample_prompt = random.choice(sample_prompt)
            prompt_inputs = tokenizer(sample_prompt)

            input_ids = prompt_inputs["input_ids"]
            attention_mask = prompt_inputs["attention_mask"]
            labels = [-100 for _ in prompt_inputs["input_ids"]]  # Default no loss for prompt

            if tokenizer_eos_token:
                text = text + tokenizer.eos_token
            text_inputs = tokenizer(text)  # Textinputs is a List[int]
            # 4. Text Tokenization and Filtering By tokens
            if exists(max_text_token_length) and len(text_inputs["input_ids"]) > max_text_token_length:
                logger.warning(
                    f"Dropped sample {data_sample.get('filename', 'unknown')} in {tarname} (max token text length)"
                )
                continue

            target_labels = text_inputs["input_ids"]

            # prompt_inputs is List[int]
            labels = labels + target_labels
            # During training, append inputs for predicting
            if append_targets_to_input:
                input_ids = input_ids + text_inputs["input_ids"]
                attention_mask = attention_mask + text_inputs["attention_mask"]

                assert len(labels) == len(input_ids) == len(attention_mask)

            yield {
                "audio": audio,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "text": text,
                "filename": filename,
            }


def url_to_name(url):
    parsed = urlparse(url)
    p = Path(parsed.path)
    filename = "_".join(p.parts[-3:])
    return filename

def download_data_to_cache(
    urls: str | List[str],
    cache_dir: str | None = None,  # Use global CACHE_DIR
    handler: Callable = warn_and_continue,
    **filtering_kwargs,
) -> wds.DataPipeline:
    def _consume_local_paths(src):
        for source in src:
            # We dont need to stream the data here, just download
            if 'stream' in source:
                del source['stream']
            yield source['url']
    return wds.DataPipeline(*[
            wds.SimpleShardList(urls),
            wds.split_by_worker,
            wds.cache.FileCache(cache_dir=cache_dir, url_to_name=url_to_name),
            _consume_local_paths,
        ])

def create_audio_text_token_pipeline(
    urls: str | List[str],
    tokenizer: Callable,
    cache_dir: str | None = None,  # Use global CACHE_DIR
    training: bool = False,
    batch_size: int = 1,
    resample: bool = False,
    handler: Callable = warn_and_continue,
    **filtering_kwargs,
) -> wds.DataPipeline:
    pipeline: List = []
    if resample:
        pipeline.append(wds.ResampledShards(urls))
    else:
        pipeline.append(wds.SimpleShardList(urls))
    # Important note: wds.tarfile_to_samples does not work for streaming
    # For streaming, one needs tar_file_expander + group_by_keys
    if training:
        pipeline.extend(
            [
                wds.detshuffle(),
                wds.split_by_node,
                wds.split_by_worker,
                wds.cache.FileCache(cache_dir=cache_dir, url_to_name=url_to_name),
                pipelinefilter(tar_file_expander)(handler=handler),
                pipelinefilter(group_by_keys)(handler=handler),
            ]
        )
    else:
        # Ensure data is distributed even if there's only one shard
        pipeline.extend(
            [
                wds.split_by_worker,
                wds.cache.FileCache(cache_dir=cache_dir, url_to_name=url_to_name),
                pipelinefilter(tar_file_expander)(handler=handler),
                pipelinefilter(group_by_keys)(handler=handler),
                wds.split_by_node,
            ]
        )
    pipeline.extend(
        [
            wds.decode(decode_torch_audio, handler=handler),
            wds.rename(
                audio="flac;mp3;sox;wav;mp4a;ogg;wma;opus",
                text="json;jsonl",
                tar="__url__",
                filename="__key__",
                handler=handler,
            ),
        ]
    )

    tokenizer_eos_token = (
        tokenizer.eos_token if (hasattr(tokenizer, "eos_token") and training) else None
    )  # Dont need to estimate EOS token during inference
    pipeline.append(
        partial(
            _process_sample_stream,
            tokenizer=tokenizer,
            tokenizer_eos_token=tokenizer_eos_token,
            append_targets_to_input=training,
            **filtering_kwargs,
        )
    )
    # Batching
    pipeline.append(
        wds.batched(
            batch_size,
            collation_fn=partial(wds.filters.default_collation_fn, combine_tensors=False, combine_scalars=False),
            partial=True,
        )
    )
    return wds.DataPipeline(*pipeline)


@dataclass
class AudioTextTokenWebdataset:
    # dataset args
    data_urls: InputUrlType
    tokenizer: PreTrainedTokenizer  # A text tokenizer
    training: bool = False
    target_sample_rate: int = 16000
    mono: bool = True
    batch_size: int = 1
    min_audio_length: float | None = None
    drop_clipped: bool = False
    normalize_clipped: bool = True
    crop_audio_length: float | None = None
    max_text_token_length: int | None = None
    resample: bool = False
    # Webloader args
    num_workers: int = 1
    sort_by_length: int | None = None
    shuffle: int = 256  # For Webloader during training
    dataset: wds.DataPipeline | None = None  # Saving the generated dataset
    cache_dir: str = ""  # use default cache dir

    def __post_init__(self):
        if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cache_dir = self.cache_dir if self.cache_dir != "" else CACHE_DIR

    def create_dataset(self) -> wds.DataPipeline:
        data_urls = parse_input_to_datatype(self.data_urls)
        datasets = []
        logger.info(
            f"Pipeline start: Training={self.training}, Resampling={self.resample}, Batch={self.batch_size}, Caching at {self.cache_dir}"
        )
        for data_type in data_urls:
            ds = create_audio_text_token_pipeline(
                urls=expand_path(data_type),
                prompt=data_type.prompt,
                text_data_key=data_type.key,
                tokenizer=self.tokenizer,
                training=self.training,
                batch_size=self.batch_size,
                target_sample_rate=self.target_sample_rate,
                mono=self.mono,
                min_audio_length=self.min_audio_length,
                drop_clipped=self.drop_clipped,
                normalize_clipped=self.normalize_clipped,
                crop_audio_length=self.crop_audio_length,
                max_text_token_length=self.max_text_token_length,
                resample=self.resample,
                cache_dir=self.cache_dir,
            )
            datasets.append((data_type, ds))

        if self.training:
            dataset = BalancedDatasetSampler(datasets=datasets)
        else:
            dataset = SequentialDatasetSampler(datasets=datasets)
        self.dataset = dataset
        return dataset

    def create_dataloader(self, collate_fn: Callable | None = None) -> wds.WebLoader:
        if self.dataset is None:
            self.create_dataset()  # Sets self.dataset
        dataloader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0) and self.training,
        ).unbatched()
        if self.training:
            dataloader = dataloader.shuffle(self.shuffle)
        if self.sort_by_length:
            dataloader = dataloader.compose(
                sort_by_length(
                    bufsize=self.sort_by_length,
                    reverse=True,
                    sort_function=lambda item: len(item["input_ids"])
                    + item["audio"].shape[-1] / 16000,  # Seconds + number of text tokens
                )
            )
        collate_fn = collate_fn or self._default_collate_fn

        dataloader = dataloader.batched(
            self.batch_size, collation_fn=collate_fn, partial=not self.training
        )  # During training remove last batch
        return dataloader

    def _default_collate_fn(self, samples):
        # Batch : Dict[str, List[List[int]]]
        input_ids, attention_mask, labels, audio = [], [], [], []
        for item in samples:
            input_ids.append(item["input_ids"])
            attention_mask.append(item["attention_mask"])
            labels.append(item["labels"])
            audio.append(item["audio"])

        padded_audio, audio_attention_mask = _pad(audio, padding_value=0.0)
        padded_batch = {
            **self.tokenizer.pad(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                padding=True,
                return_tensors="pt",
            ),  # input_ids + attention_mask
            "labels": _pad(labels, padding_value=-100)[0],
            "audio": padded_audio,
            "audio_attention_mask": audio_attention_mask,
        }

        padded_batch["audio_attention_mask"] = audio_attention_mask
        padded_batch["audio"] = padded_audio
        return padded_batch


class BalancedDatasetSampler(wds.mix.RandomMix):
    def __init__(self, datasets: List[Tuple[AudioTextDataType, Iterable]]):
        given_probs = [d[0].prob for d in datasets]
        data_tar_lengths = np.array([len(d[1].pipeline[0].urls) for d in datasets])
        normalized_tar_probs = data_tar_lengths / sum(data_tar_lengths)
        # If given prob is provided use the probability otherwise, sample by number of tars
        probs = [p if p is not None else float(orig_p) for p, orig_p in zip(given_probs, normalized_tar_probs)]
        normalized_probs = [p / sum(probs) for p in probs]
        for prob, dname in zip(normalized_probs, [d[0].name for d in datasets]):
            logger.debug(f"Sampling {dname or '':>35}: {prob:.3%}")
        super().__init__(datasets=[d[1] for d in datasets], probs=probs, longest=True)


class SequentialDatasetSampler(wds.DataPipeline, wds.compat.FluidInterface):
    def __init__(self, datasets: List[tuple[AudioTextDataType, Iterable]]):
        super().__init__()
        self.datasets = [d[1] for d in datasets]

    def __iter__(self):
        sources = [iter(ds) for ds in self.datasets]
        for source in sources:
            yield from source


# def expand_path(pattern: str | List[str]) -> List[str]:
def expand_path(data_type: AudioTextDataType) -> List[str]:
    import braceexpand, glob

    pattern = data_type.data

    if isinstance(pattern, str):
        pattern_list = [pattern]
    elif isinstance(pattern, list):
        pattern_list = pattern
    else:
        return []
    all_final_matches: List[str] = []

    # Passed a hf dataset
    if data_type.repo_id is not None:
        for pattern in pattern_list:
            for url in braceexpand.braceexpand(pattern):
                all_final_matches.append(resolve_url(url, repo_id=data_type.repo_id))
    else:
        for pattern in pattern_list:
            scheme = urlparse(pattern).scheme
            if scheme != "":  # Is HTTPS or internet-y, Just use the provided link directly
                for url in braceexpand.braceexpand(pattern):
                    all_final_matches.append(url)
            else:
                intermediate_patterns: List[str] = list(braceexpand.braceexpand(pattern))
                has_glob_chars = any(c in pattern for c in "*?[]")

                for p in intermediate_patterns:
                    expanded_p = os.path.expanduser(p)
                    matches = glob.glob(expanded_p)
                    if not matches and not has_glob_chars:
                        all_final_matches.append(expanded_p)
                    else:
                        all_final_matches.extend(matches)
    return sorted(list(set(all_final_matches)))


if __name__ == "__main__":
    from tqdm import tqdm
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = AudioTextTokenWebdataset(
        tokenizer=tokenizer,
        data_urls=[
            "https://hf-mirror.com/datasets/mispeech/MECAT-Caption/resolve/main/SM0/train_0000-0000000.tar.gz?download=true",
            # 'env/AISHELL-1/train/SLR33_Aishell1_179h_0000*.tar.gz'
        ],
        batch_size=128,
        num_workers=0,
    ).create_dataloader()
    for _ in tqdm(ds):
        print(_["audio"].shape)
        pass
