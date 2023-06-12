import argparse
import copy
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn

from utilities import GhostDataset, training_loop, testing_loop
from networks import GhostNetwork


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        help="NumPy base filename containing dataset",
        type=str,
        required=True,
    )
    parser.add_argument("--nocuda", help="Disable CUDA", action="store_true")
    # parameters
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=1024)
    parser.add_argument("--batch", help="Batch size", type=int, default=512)
    parser.add_argument("--learning", help="Learning rate", type=float, default=1e-3)
    # misc
    parser.add_argument("-v", "--verbose", help="Verbose", action="store_true")
    parser.add_argument(
        "--int8", help="Quantize the trained model to INT8", action="store_true"
    )
    parser.add_argument("--plot", help="Plot accuracy over time", action="store_true")
    parser.add_argument(
        "--save", help="Save the trained model to disk", action="store_true"
    )
    return parser.parse_args()


def __main__():
    arguments = command_line()
    if not arguments.nocuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # create training, validation, and testing data sets
    data_train = np.load(f"{arguments.filename}_train_data.npy")
    labels_train = np.load(f"{arguments.filename}_train_labels.npy")
    print(f"Training set size: {len(data_train)}")
    data_validation = np.load(f"{arguments.filename}_valid_data.npy")
    labels_validation = np.load(f"{arguments.filename}_valid_labels.npy")
    print(f"Validation set size: {len(data_validation)}")
    data_test = np.load(f"{arguments.filename}_test_data.npy")
    labels_test = np.load(f"{arguments.filename}_test_labels.npy")
    print(f"Test set size: {len(data_test)}")
    training_dataset = GhostDataset(
        torch.tensor(data_train, dtype=torch.float32, device=device),
        torch.tensor(labels_train, dtype=torch.float32, device=device),
    )
    validation_dataset = GhostDataset(
        torch.tensor(data_validation, dtype=torch.float32, device=device),
        torch.tensor(labels_validation, dtype=torch.float32, device=device),
    )
    test_dataset = GhostDataset(
        torch.tensor(data_test, dtype=torch.float32, device=device),
        torch.tensor(labels_test, dtype=torch.float32, device=device),
    )
    training_dataloader = DataLoader(training_dataset, batch_size=arguments.batch)
    validation_dataloader = DataLoader(validation_dataset, batch_size=arguments.batch)
    test_dataloader = DataLoader(test_dataset, batch_size=arguments.batch)
    # model
    num_features = data_train.shape[1]
    model = GhostNetwork(num_features, l0=int(num_features * 2.5))
    print(f"Device: {device}")
    model.to(device)
    print()
    print(model)
    print(
        f"Model parameters: {sum([x.reshape(-1).shape[0] for x in model.parameters()])}"
    )
    print()
    # training and testing
    num_epochs = arguments.epochs
    batch_size = arguments.batch
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning)
    best_accuracy = -np.inf
    accuracy_history = list()
    loss_history = list()
    best_weights = None
    start_time = perf_counter()
    for epoch in range(0, num_epochs):
        if arguments.verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}")
        training_loop(model, training_dataloader, loss_function, optimizer)
        accuracy, loss = testing_loop(model, validation_dataloader, loss_function)
        accuracy_history.append(accuracy * 100.0)
        loss_history.append(loss)
        if arguments.verbose:
            print(f"\tAccuracy: {accuracy * 100.0:.2f}%")
            print(f"\tLoss: {loss:.6f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = copy.deepcopy(model.state_dict())
    end_time = perf_counter()
    model.load_state_dict(best_weights)
    print()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Best Accuracy: {best_accuracy * 100.0:.2f}%")
    print()
    accuracy, loss = testing_loop(model, test_dataloader, loss_function)
    print(f"Test Accuracy: {accuracy * 100.0:.2f}%")
    print(f"Test Loss: {loss:.6f}")
    print()
    # plotting
    if arguments.plot:
        epochs = np.arange(0, num_epochs)
        plt.plot(epochs, accuracy_history, "r", label="Validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.legend(loc="lower right")
        plt.show()
        plt.plot(epochs, loss_history, "r", label="Validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Loss")
        plt.legend(loc="upper right")
        plt.show()
    # save model
    if arguments.save:
        print("Saving model to disk")
        torch.save(model.state_dict(), "ghost_model.pth")
        print("Saving model to ONNX format")
        dummy_input = torch.randn(1, num_features)
        torch.onnx.export(model, dummy_input, "ghost_model.onnx", export_params=True)
    # INT8 quantization
    if arguments.int8:
        print("INT8 quantization")
        model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        model_fused = torch.quantization.fuse_modules(model, [["layer0", "relu"]])
        model_prepared = torch.quantization.prepare_qat(model_fused.train())
        for epoch in range(0, num_epochs):
            training_loop(model_prepared, training_dataloader, loss_function, optimizer)
        model_prepared.eval()
        model_int8 = torch.quantization.convert(model_prepared)
        print()
        print(model_int8)
        print()
        # save model
        if arguments.save:
            print("Saving INT8 model to disk")
            torch.save(model_int8, "ghost_model_int8.pth")
            print("Saving INT8 model to ONNX format")
            dummy_input = torch.randn(1, num_features)
            torch.onnx.export(
                model_int8, dummy_input, "ghost_model_int8.onnx", export_params=True
            )


if __name__ == "__main__":
    __main__()
