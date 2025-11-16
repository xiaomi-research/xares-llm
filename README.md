# XARES-LLM

XARES-LLM trains a typical LALM using the audio encoder provided by the user. 
The system automatically downloads training data, trains the LALM then tests various downstream tasks, providing scores for each. 

<img src="xares-llm.png" alt="framework" width="85%">

## Installation

```bash
uv venv 
uv pip install $THIS_REPO
```

## Usage


You can either pass a .py file with your encoder defined inside, requiring the ending "Encoder":

```bash
python3 -m xares_llm.run example/dummy/dummyencoder.py 
```

Or pass the module directly:

```bash
python3 -m xares_llm.run example.dummy.dummyencoder.DummyEncoder
```

### Single dataset training


Here we train on clotho and test on clotho.

```bash
python3 -m xares_llm.run example/dummy/dummyencoder.py clotho clotho
# Or for the Module interface
# python3 -m xares_llm.run example.dummy.dummyencoder.DummyEncoder clotho clotho

```


All available current datasets can be seen by running `python3 -m xares_llm.run -h`.


Datasets can also be passed with a custom .yaml:

For training, the format is:

```yaml
train_data:
  CustomDataName:
    prompt: My prompt
    data:
    - PATH_TO_MY_TARS{00..10}.tar
    key: DATAKEY
num_training_workers: 4
```


For evaluation:

```yaml
eval_custom:
  data:
    data:
    - PATH_TO_MY_TARS{00..10}.tar
    key: DATAKEY # Inside the json
    prompt: My prompt
  batch_size: 4
  num_workers: 0
  metric: Accuracy
```


### Modify dataset

By default all data is downloaded and stored in `xares_data` from the current directory.
One can modify the data path with the environment variable `XARES_DATA_HOME`.



