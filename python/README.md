
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

All the packages except ROOT are available via PyPI and can be installed with `python -m pip install <package>`.
ROOT can be installed via Conda following [these instructions](https://iscinumpy.gitlab.io/post/root-conda/).

## Usage

### Create datasets from ROOT file

Before training we create a dataset (NumPy arrays) from the ROOT file. To create the dataset you can use `make_dataset.py`.

Example

```bash
python make_dataset.py -f ../data/Data.root --output ../data/dataset
```

The script will create 6 files in the `../data` directory. The files are:

* `dataset_train_data`
* `dataset_train_labels`
* `dataset_valid_data`
* `dataset_valid_labels`
* `dataset_test_data`
* `dataset_test_labels`

### Training

The training process includes hyperparameters tuning.

Example

```bash
python training.py -f ../data/dataset --save
```

The script will produce 3 files:

* `ghost_model.pth`: the weights for the PyTorch model
* `ghost_model_config.pkl`: the best configuration of the hyperparameters
* `ghost_model.oonx`: weights and model in the ONNX format

### Inference and Analysis

The inference script runs the inference, perform threshold analysis (the threshold is used to decide if a track is a ghost or not), and plots various statistics on the data.

Example

```bash
python inference.py -f ../data/Data.root --model ghost_model.pth --config ghost_model_config.pkl
```
