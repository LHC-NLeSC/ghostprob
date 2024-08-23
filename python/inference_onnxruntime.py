import argparse
from time import perf_counter
import numpy as np
import torch
from torch.utils.data import DataLoader
import onnxruntime as ort
import matplotlib.pyplot as plt

from utilities import load_data, remove_nans, shuffle_data, normalize
from data import (
    label,
    training_columns_forward,
    training_columns_matching,
    GhostDataset,
)

thresholds = [
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.81,
    0.82,
    0.83,
    0.84,
    0.85,
    0.86,
    0.87,
    0.88,
    0.89,
    0.9,
    0.95,
]


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
    parser.add_argument(
        "--normalize",
        help="Normalize input data before inference.",
        action="store_true",
    )
    parser.add_argument(
        "--track",
        help="Forward or Matching",
        type=str,
        choices=["forward", "matching"],
        required=True,
    )
    parser.add_argument("--batch", help="Batch size", type=int, default=1)
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
        if arguments.track.lower() == "forward":
            training_columns = training_columns_forward
        elif arguments.track.lower() == "matching":
            training_columns = training_columns_matching
        for column in training_columns:
            if column not in columns:
                print("Missing data.")
                return
        # create dataset
        data = [dataframe[column] for column in training_columns]
        # Remove NaNs
        data, labels = remove_nans(data, labels)
        # Normalize
        if arguments.normalize:
            for feature_id in range(len(data)):
                data[feature_id] = normalize(data[feature_id])
        # Shuffle
        data = np.hstack([data[i].reshape(len(data[0]), 1) for i in range(len(data))])
        rng = np.random.default_rng()
        data, labels = shuffle_data(rng, data, labels)
    else:
        data = np.load(f"{arguments.filename}_test_data.npy")
        labels = np.load(f"{arguments.filename}_test_labels.npy")
        print(f"Test set size: {len(data)}")
    tracks = len(data)
    values, counts = np.unique(labels, return_counts=True)
    p_ghosts = dict(zip(values, counts))[1] / tracks
    p_real = dict(zip(values, counts))[0] / tracks
    test_dataset = GhostDataset(
        torch.tensor(data, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    # read model and initialize ONNXRuntime
    ort_session = ort.InferenceSession(arguments.model)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    test_dataloader = DataLoader(test_dataset, batch_size=arguments.batch, shuffle=True)
    # Accuracy test (CLI)
    accuracies = list()
    for threshold in thresholds:
        start_time = perf_counter()
        accuracy = [0, 0]
        for x, y in test_dataloader:
            prediction = ort_session.run([output_name], {input_name: x.numpy()})[0]
            prediction = prediction > threshold
            for i in range(0, len(prediction)):
                # True Positive
                if prediction[i] and y.int()[i] == 1:
                    accuracy[0] += 1
                # True Negative
                elif not prediction[i] and y.int()[i] == 0:
                    accuracy[1] += 1
        temp_accuracy = ((accuracy[0] + accuracy[1]) * 100) / len(data)
        end_time = perf_counter()
        accuracies.append(temp_accuracy)
        print(f"Threshold: {threshold}")
        print(f"\tAccuracy: {temp_accuracy:.2f}%")
        print(f"\tInference time: {end_time - start_time:.2f} seconds")
        print()
    print(f"Tracks: {p_real * 100.0:.2f}%")
    print(f"Ghosts: {p_ghosts * 100.0:.2f}%")
    print()
    # Plot accuracy over threshold
    plt.plot(thresholds, accuracies, label="Accuracy")
    plt.xlabel("Threshold")
    plt.xticks(thresholds)
    plt.ylim(0, 100)
    plt.legend()
    plt.show()
    # Plot accuracy components
    tp = list()
    tn = list()
    fp = list()
    fn = list()
    for threshold in thresholds:
        accuracy = [0, 0, 0, 0]
        for x, y in test_dataloader:
            prediction = ort_session.run([output_name], {input_name: x.numpy()})[0]
            prediction = prediction > threshold
            for i in range(0, len(prediction)):
                # True Positive
                if prediction[i] and y.int()[i] == 1:
                    accuracy[0] += 1
                # True Negative
                elif not prediction[i] and y.int()[i] == 0:
                    accuracy[1] += 1
                # False Positive
                elif prediction[i] and y.int()[i] == 0:
                    accuracy[2] += 1
                # False Negative
                elif not prediction[i] and y.int()[i] == 1:
                    accuracy[3] += 1
        tp.append(accuracy[0])
        tn.append(accuracy[1])
        fp.append(accuracy[2])
        fn.append(accuracy[3])
    g_tp = list()
    g_tn = list()
    g_fp = list()
    g_fn = list()
    nn_ghost = list()
    nn_real = list()
    sensitivity = list()
    specificity = list()
    fnr = list()
    fpr = list()
    ppv = list()
    for i in range(0, len(thresholds)):
        print(f"Threshold {thresholds[i]}")
        g_tp.append(tp[i] / tracks)
        g_tn.append(tn[i] / tracks)
        g_fp.append(fp[i] / tracks)
        g_fn.append(fn[i] / tracks)
        nn_ghost.append((tp[i] + fp[i]) / tracks)
        nn_real.append((tn[i] + fn[i]) / tracks)
        sensitivity.append(tp[i] / (tp[i] + fn[i]))
        specificity.append(tn[i] / (tn[i] + fp[i]))
        fnr.append(fn[i] / (fn[i] + tp[i]))
        fpr.append(fp[i] / (fp[i] + tn[i]))
        ppv.append(tp[i] / (tp[i] + fp[i]))
        print(f"\tTracks: {nn_real[i] * 100.0:.2f}%")
        print(f"\tGhosts: {nn_ghost[i] * 100.0:.2f}%")
        print()
        print(f"\tTP: {g_tp[i] * 100.0:.2f}%")
        print(f"\tTN: {g_tn[i] * 100.0:.2f}%")
        print(f"\tFP: {g_fp[i] * 100.0:.2f}%")
        print(f"\tFN: {g_fn[i] * 100.0:.2f}%")
        print()
        print(f"\tSensitivity: {sensitivity[i] * 100.0:.2f}%")
        print(f"\tSpecificity: {specificity[i] * 100.0:.2f}%")
        print(f"\tFalse Positive Rate: {fpr[i] * 100.0:.2f}%")
        print(f"\tFalse Negative Rate: {fnr[i] * 100.0:.2f}%")
        print()
    plt.plot(thresholds, [p_ghosts] * len(thresholds), label="Ghost Tracks")
    plt.plot(thresholds, [p_real] * len(thresholds), label="Real Tracks")
    plt.plot(thresholds, nn_ghost, label="NN - Ghost Tracks")
    plt.plot(thresholds, nn_real, label="NN - Real Tracks")
    plt.xticks(thresholds)
    plt.legend()
    plt.show()
    plt.plot(thresholds, g_tp, label="True Positives")
    plt.plot(thresholds, g_tn, label="True Negatives")
    plt.plot(thresholds, g_fp, label="False Positives")
    plt.plot(thresholds, g_fn, label="False Negatives")
    plt.xticks(thresholds)
    plt.legend()
    plt.show()
    plt.plot(thresholds, sensitivity, label="Sensitivity")
    plt.plot(thresholds, specificity, label="Specificity")
    plt.plot(thresholds, fnr, label="False Negative Rate")
    plt.plot(thresholds, fpr, label="False Positive Rate")
    plt.xticks(thresholds)
    plt.legend()
    plt.show()
    plt.plot(fpr, sensitivity, label="Neural Network")
    plt.plot([0, 0.5, 1], [0, 0.5, 1], label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    J = np.asarray(sensitivity) - np.asarray(fpr)
    f1_score = (np.asarray(tp) * 2) / (
        np.asarray(tp) * 2 + np.asarray(fp) + np.asarray(fn)
    )
    print(f"Best threshold (J): {thresholds[np.argmax(J)]}")
    print(f"Best threshold (F1): {thresholds[np.argmax(f1_score)]}")


if __name__ == "__main__":
    __main__()
