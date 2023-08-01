import argparse
import pickle
import numpy as np
import torch
import onnxruntime as ort

from utilities import load_data, remove_nans, shuffle_data, GhostDataset, DataLoader
from data import label, training_columns


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filename", help="File containing the data set", type=str, required=True
    )
    parser.add_argument(
        "--model",
        help="Name of the file containing the model.",
        type=str,
        required=True,
    )
    parser.add_argument("--threshold", help="Ghost threshold.", type=float, default=0.5)
    return parser.parse_args()



def __main__():
    arguments = command_line()
    if ".root" in arguments.filename:
        dataframe, columns = load_data(arguments.filename)
        print(f"Columns in the table: {len(dataframe)}")
        print(columns)
        if label not in columns:
            print("Missing labels.")
            return
        labels = dataframe[label].astype(int)
        for column in training_columns:
            if column not in columns:
                print("Missing data.")
                return
        # create dataset
        data = [dataframe[column] for column in training_columns]
        # Remove NaNs
        data, labels = remove_nans(data, labels)
        # split into ghost and real tracks
        data = np.hstack([data[i].reshape(len(data[0]), 1) for i in range(len(data))])
        data_ghost = data[labels == 1]
        data_real = data[labels == 0]
        print(
            f"Number of ghost ({len(data_ghost)}) and real tracks ({len(data_real)}) in data set"
        )
        # select the same number of other real tracks as there are ghosts
        rng = np.random.default_rng()
        rng.shuffle(data_real)
        data_real = data_real[: len(data_ghost)]
        # create data set
        data = np.vstack((data_ghost, data_real))
        labels_ghost = np.ones((len(data_ghost), 1), dtype=int)
        labels_real = np.zeros((len(data_real), 1), dtype=int)
        labels = np.vstack((labels_ghost, labels_real))
        data, labels = shuffle_data(rng, data, labels)
    else:
        data = np.load(f"{arguments.filename}_test_data.npy")
        labels = np.load(f"{arguments.filename}_test_labels.npy")
        print(f"Test set size: {len(data)}")
    print()
    test_dataset = GhostDataset(
        torch.tensor(data, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    # read model and initialize ONNXRuntime
    ort_session = ort.InferenceSession(arguments.model)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    # run inference
    # TP, TN, FP, FN
    accuracy = [0, 0, 0, 0]
    for i in range(len(test_dataset)):
        x, y  = test_dataset[i]
        prediction = ort_session.run([output_name], {input_name: [x.numpy()]})[0][0][0]
        prediction = int((prediction > arguments.threshold))
        if prediction == 1 and y.int() == 1:
            accuracy[0] += 1
        # True Negative
        elif prediction == 0 and y.int() == 0:
            accuracy[1] += 1
        # False Positive
        elif prediction == 1 and y.int() == 0:
            accuracy[2] += 1
        # False Negative
        elif prediction == 0 and y.int() == 1:
            accuracy[3] += 1
    print(f"True positive: {accuracy[0]}")
    print(f"True negative: {accuracy[1]}")
    print(f"False positive: {accuracy[2]}")
    print(f"False negative: {accuracy[3]}")
    print()
    print(f"Accuracy: {((accuracy[0] + accuracy[1])/(np.sum(accuracy)) * 100):.2f}%")

if __name__ == "__main__":
    __main__()
