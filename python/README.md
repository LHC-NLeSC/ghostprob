
# Ghost Tracks

Python and PyTorch scripts to train a neural network able to recognize if a reconstructed track is a ghost or not.

## Dependencies

The following Python packages are necessary to use the scripts:

* numpy
* torch
* onnx
* onnx2torch
* matplotlib
* ROOT
* ray

All the packages except ROOT are available via PyPI and can be installed with `python -m pip install <package>`.
ROOT can be installed via Conda following [these instructions](https://iscinumpy.gitlab.io/post/root-conda/).

## Usage

### Create datasets from ROOT file

Before training we create a dataset (NumPy arrays) from the ROOT file. To create the dataset you can use `make_dataset.py`.

#### Example

```bash
python make_dataset.py -f ../data/Data.root --output ../data/dataset --track matching
```

The script will create 6 files in the `../data` directory with the `dataset` prefix. The files are:

* `dataset_train_data`
* `dataset_train_labels`
* `dataset_valid_data`
* `dataset_valid_labels`
* `dataset_test_data`
* `dataset_test_labels`

The `--track` argument can be either "`forward`" or "`matching`" and it is necessary to know which features include in the dataset.

#### Optional arguments

* `--fraction <float>` to only use a fraction of the data
* `--plot` to plot an histogram of each feature during the dataset creation
* `--normalize` to normalize the features in the `[0, 1]` set (necessary in case you want to use a network that does not do normalization)

### Training

The training process includes hyperparameters tuning using `ray`.

#### Example

```bash
python training.py -f ../data/dataset --save
```

The script will produce 3 files:

* `ghost_model.pth`: the weights for the PyTorch model
* `ghost_model_config.pkl`: the best configuration of the hyperparameters
* `ghost_model.onnx`: weights and model in the ONNX format

#### Optional arguments

* `--network [0, 1]` to train one of the two available networks (`0` is the network without normalization and `1` is the network with normalization)
* `--epochs <int>` the number of epochs
* `-n <int>` the number of configurations to explore during tuning
* `--threshold <float>` the threshold for classifying a track as a ghost in the interval `[0, 1]`
* `--batch <int>` the batch size for training
* `--path <string>` a directory to store the output of `ray`
* `--cpu <int>` number of cores to use
* `--gpu <int>` number of GPUs to use
* `--cuda <int>` the CUDA device to use
* `--nocuda` disable CUDA
* `--int8` use INT8 quantization

### Inference and Analysis

The inference script runs the inference, perform threshold analysis (the threshold is used to decide if a track is a ghost or not), and plots various statistics on the data.

#### Example

```bash
python inference.py -f ../data/Data.root --model ghost_model.onnx --config ghost_model_config.pkl --track matching
```

The `--track` argument can be either "`forward`" or "`matching`" and it is necessary to know which features include in the dataset.

#### Optional arguments

* `--nocuda` disable CUDA
* `--network [0, 1]` to train one of the two available networks (`0` is the network without normalization and `1` is the network with normalization)
* `--normalize` normalize input data before inference in the set `[0, 1]` (necessary if using a network withouth a normalization layer)
* `--int8` use INT8 quantization