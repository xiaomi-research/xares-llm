# XARES-LLM

XARES-LLM 2026 Challenge baseline and evaluation code.

## Installation

```bash
uv venv 
uv pip install $THIS_REPO
```

## Usage

```bash
python3 -m xares_llm.run example/dummy/dummyencoder.py src/xares_llm/tasks/single/fsdkaggle2018.yaml src/xares_llm/tasks/evaluation/eval_fsdkaggle2018.yaml 
```



### Modify dataset

By default all data is downloaded and stored in `xares_data` from the current directory.
One can modify the data path with the environment variable `XARES_DATA_HOME`.


