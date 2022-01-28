import math
import numpy as np
import tensorflow as tf
from ROOT import TFile, RDataFrame

kalman_file = TFile("../build-cpu/output/KalmanIPCheckerOutput.root")
columns = ["x", "y", "tx", "ty", "best_qop", "best_pt", "kalman_ip_chi2",
           "kalman_docaz", "chi2", "chi2V", "chi2UT", "chi2T",
           "ndof", "ndofV", "ndofT", "nUT", "ghost"]
df = RDataFrame("kalman_ip_tree", kalman_file, columns).Define("p", "abs(1.f/best_qop)")
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


def norm(x):
    return x / math.sqrt(1. + x * x)


norm_v = np.vectorize(norm)

# Scale data to [-1., 1.]
# data_columns = [norm_v(c) for c in data_columns]

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

loss_fn = tf.keras.losses.BinaryCrossentropy()

opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(data_train, labels_train, batch_size=1000, epochs=1000)

model.evaluate(data_test, labels_test, verbose=2)
