import argparse
from functools import partial
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

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
    parser.add_argument("-n", "--num_samples", help="Samples for hyperparameter tuning.", type=int, default=128)
    # misc
    parser.add_argument(
        "--int8", help="Quantize the trained model to INT8", action="store_true"
    )
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
    print(f"Device: {device}")
    # initialize ray
    ray.init(logging_level=logging.CRITICAL)
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
    # Training and tuning hyperparameters
    num_features = data_train.shape[1]
    num_epochs = arguments.epochs
    tuning_config = {
        "l0": tune.choice(
            [i for i in range(int(num_features / 2), int(num_features * 10))]
        ),
        "learning": tune.loguniform(1e-6, 1),
        "batch": tune.choice([2**i for i in range(1, 15)]),
        "epochs": tune.choice([num_epochs]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    loss_function = nn.BCELoss()
    result = tune.run(
        partial(training_loop, num_features=num_features,
            device=device,
            loss_function=loss_function,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset),
        config=tuning_config,
        num_samples=arguments.num_samples,
        scheduler=scheduler,
        storage_path="./ray_logs",
        checkpoint_score_attr="loss",
        progress_reporter=reporter,
    )
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    # Test accuracy
    test_dataloader = DataLoader(test_dataset, batch_size=best_trial.config["batch"])
    model = GhostNetwork(num_features, l0=best_trial.config["l0"])
    model.to(device)
    print()
    print(model)
    print(
        f"Model parameters: {sum([x.reshape(-1).shape[0] for x in model.parameters()])}"
    )
    print()
    accuracy, loss = testing_loop(model, test_dataloader, loss_function)
    print(f"Test Accuracy: {accuracy * 100.0:.2f}%")
    print(f"Test Loss: {loss:.6f}")
    print()
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
            training_loop()
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
