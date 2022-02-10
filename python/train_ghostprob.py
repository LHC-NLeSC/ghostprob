import argparse
import math
import numpy as np
import tensorflow as tf
import onnx
import tf2onnx

from os import environ
from ROOT import TFile, RDataFrame

environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="File with validator data", type=str, required=True)
    # Parameters
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=512)
    parser.add_argument("--batch", help="Batch size", type=int, default=1024)
    # Preprocessing
    parser.add_argument("--bound", help="Filter entries outside the boundaries", action="store_true")
    parser.add_argument("--normalize", help="Use a normalization layer", action="store_true")
    # Analysis
    parser.add_argument("--plot", help="Plot accuracy over time", action="store_true")
    parser.add_argument("--save", help="Save the trained model to disk", action="store_true")
    return parser.parse_args()


def norm(x):
    return x / math.sqrt(1. + x * x)


arguments = command_line()

kalman_file = TFile(arguments.filename)
columns = ["x", "y", "tx", "ty", "best_qop", "best_pt", "kalman_ip_chi2",
           "kalman_docaz", "chi2", "chi2V", "chi2UT", "chi2T",
           "ndof", "ndofV", "ndofT", "nUT", "ghost"]
df = RDataFrame("kalman_validator/kalman_ip_tree", kalman_file, columns).Define("p", "abs(1.f/best_qop)")

bounds = {"x": (-10., 10.),
          "y": (-10., 10.),
          "tx": (-0.3, 0.3),
          "ty": (-0.3, 0.3),
          "best_pt": (0, 15000),
          "kalman_ip_chi2": (-0.5, 10000.5),
          "kalman_docaz": (-0.5, 25.5),
          "chi2dof": (0, 400),
          "chi2Vdof": (0, 150),
          "chi2Tdof": (0, 150)}


if arguments.bound:
    for column in columns:
        if column in bounds:
            lower, upper = bounds[column]
            df = df.Filter("{0} > {1} && {0} < {2}".format(column, lower, upper))

np_df = df.AsNumpy()

labels_all = np_df['ghost'].astype(int)

columns = ["p"] + columns[:-1]

data_columns = [np_df[c] for c in columns]

# Remove NaNs
for i_col, column in enumerate(data_columns):
    index = np.isfinite(column)
    if len(np.unique(index)) == 2:
        for j_col in range(len(data_columns)):
            data_columns[j_col] = data_columns[j_col][index]
        labels_all = labels_all[index]


if arguments.normalize:
    norm_v = np.vectorize(norm)
    # Scale data to [-1., 1.]
    data_columns = [norm_v(c) for c in data_columns]

n_rows = len(data_columns[0])

data_all = np.hstack([data_columns[i].reshape(n_rows, 1) for i in range(len(data_columns))])

# Split into data for ghosts and tracks to get a better mix
data_ghosts = data_all[labels_all == 1]
data_tracks = data_all[labels_all == 0]

n_ghosts = data_ghosts.shape[0]

# 50/50 ghosts and tracks
data_tracks = data_tracks[:n_ghosts]

data = np.vstack((data_tracks, data_ghosts))

# Create labels for tracks and ghosts
labels_tracks = np.zeros((n_ghosts, 1), dtype=int)
labels_ghosts = np.ones((n_ghosts, 1), dtype=int)

labels = np.vstack((labels_tracks, labels_ghosts))


# Shuffle tracks data to have a mixed sample
def shuffled_data(data, labels):
    assert(len(data) == len(labels))
    p = np.random.permutation(len(data))
    return data[p], labels[p]


data, labels = shuffled_data(data, labels)


split = int(len(data) / 4)
data_test = data[:split]
data_train = data[split:]
labels_test = labels[:split]
labels_train = labels[split:]

n_input = data.shape[1]
n_hidden = int(1.5 * n_input)
n_output = 1

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(n_hidden, input_dim=n_input, activation='relu'),
  tf.keras.layers.Dense(n_hidden - 5, activation='relu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(n_output, activation='sigmoid')
])
print()
model.summary()
print()

loss_fn = tf.keras.losses.BinaryCrossentropy()

opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

training_history = model.fit(data_train, labels_train, batch_size=1000, epochs=1000, verbose=0)

loss, accuracy = model.evaluate(data_test, labels_test, verbose=0)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Plotting
if arguments.plot:
    epochs = np.arange(0, num_epochs)
    plt.plot(epochs, training_history.history["loss"], "bo", label="Training loss")
    plt.plot(epochs, training_history.history["val_loss"], "ro", label="Validation loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")
    plt.show()
    plt.plot(epochs, training_history.history["accuracy"], "b", label="Training accuracy")
    plt.plot(epochs, training_history.history["val_accuracy"], "r", label="Validation accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.show()

if arguments.save:
    print("Saving model to disk")
    model.save("train_ghostprob_model.h5")
    print("Saving model to ONNX format")
    input_signature = [tf.TensorSpec(input.shape, input.dtype) for input in model.inputs]
    model_onnx, _ = tf2onnx.convert.from_keras(model, input_signature)
    onnx.save(model_onnx, "train_ghostprob_model.onnx")
