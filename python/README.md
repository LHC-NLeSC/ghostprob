
# Ghost Tracks

Python and tensorflow scripts to train a neural network able to recognize if a reconstructed track is a ghost or not.

## Dependencies

The following Python packages are necessary to use the scripts:

* numpy
* tensorflow
* onnx
* tf2onnx
* matplotlib
* ROOT

All the packages except ROOT are available via PyPI and can be installed with `python -m pip install <package>`.
ROOT can be installed via Conda following [these instructions](https://iscinumpy.gitlab.io/post/root-conda/).


## train_ghostprob.py

The script has a helper to show how to call it, executing `python train_ghostprob.py` produces the following output:

```
usage: train_ghostprob.py [-h] -f FILENAME [--epochs EPOCHS] [--batch BATCH]
                          [--bound] [--normalize] [--plot] [--save]
train_ghostprob.py: error: the following arguments are required: -f/--filename
```

The parameters are the following:
* `-f` is the ROOT filename containing the simulation data
* `--epochs` is the number of epochs for the training
* `--batch` the batch size for training
* `--bound` to enable bounding of input variables
* `--normalize` to enable input normalization
* `--plot` to plot the value of loss function and accuracy during training
* `--save` to save the trained model to disk


## ghostprob.py

The script has a helper to show how to call it, executing `python ghostprob.py` produces the following output:

```
usage: ghostprob.py [-h] -f FILENAME [--epochs EPOCHS] [--batch BATCH]
                    [--bound] [--normalize] [--plot] [--save]
ghostprob.py: error: the following arguments are required: -f/--filename
```

The parameters are the following:
* `-f` is the ROOT filename containing the simulation data
* `--epochs` is the number of epochs for the training
* `--batch` the batch size for training
* `--bound` to enable bounding of some input variables
* `--normalize` to enable input normalization
* `--plot` to plot the value of loss function and accuracy during training
* `--save` to save the trained model to disk

