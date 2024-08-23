import argparse
import logging
import pickle
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler

from utilities import (
    training_loop,
    inner_training_loop,
    testing_loop,
    select_optimizer,
)
from networks import (
    GhostNetwork,
    GhostNetworkWithNormalization,
    GhostNetworkWithManualNormalization,
)
from data import GhostDataset


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        help="NumPy base filename containing dataset",
        type=str,
        required=True,
    )
    # parameters
    parser.add_argument(
        "--network", help="Network to train", type=int, choices=range(0, 3), default=0
    )
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=256)
    parser.add_argument(
        "-n",
        "--num_samples",
        help="Samples for hyperparameter tuning.",
        type=int,
        default=128,
    )
    parser.add_argument("--threshold", help="Ghost threshold.", type=float, default=0.5)
    parser.add_argument("--batch", help="Batch size.", type=int, default=2048)
    # misc
    parser.add_argument(
        "--path",
        help="Where to store the tuning output.",
        type=str,
        default="/tmp/ghostprob/",
    )
    parser.add_argument(
        "--tmp-path",
        help="Where to store the tuning output.",
        type=str,
        default="/tmp/ray/",
    )
    parser.add_argument(
        "--cpu", help="Number of CPU cores to use for training.", type=int, default=1
    )
    parser.add_argument("--nocuda", help="Disable CUDA", action="store_true")
    parser.add_argument("--cuda", help="ID of the CUDA device.", type=int, default=0)
    parser.add_argument(
        "--gpu", help="Number of GPUs to use for training.", type=int, default=0
    )
    parser.add_argument(
        "--int8", help="Quantize the trained model to INT8.", action="store_true"
    )
    parser.add_argument(
        "--save", help="Save the trained model to disk", action="store_true"
    )
    return parser.parse_args()


def __main__():
    arguments = command_line()
    use_cuda = not arguments.nocuda and torch.cuda.is_available()
    # initialize ray
    ray.init(
        num_cpus=arguments.cpu,
        num_gpus=arguments.gpu,
        configure_logging=True,
        log_to_driver=False,
        logging_level=logging.ERROR,
        include_dashboard=False,
        _temp_dir=arguments.tmp_path,
    )
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
    num_features = data_train.shape[1]
    num_epochs = arguments.epochs
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=num_epochs,
        grace_period=8,
        reduction_factor=2,
    )
    loss_function = nn.BCELoss()
    # search space
    tuning_config = {
        "num_features": num_features,
        "threshold": arguments.threshold,
        "l0": tune.choice(
            [i for i in range(int(num_features / 3), int(num_features * 3), 4)]
        ),
        "learning": tune.loguniform(1e-6, 1e-1),
        "batch": arguments.batch,
        "epochs": num_epochs,
        "optimizer": tune.choice([0, 1]),
        "activation": tune.choice(
            [
                nn.ReLU,
                nn.Tanh,
                nn.Sigmoid,
                nn.LeakyReLU,
                nn.ELU,
                nn.Softmax,
                nn.Softmin,
            ]
        ),
        "data_train": data_train,
        "labels_train": labels_train,
        "data_validation": data_validation,
        "labels_validation": labels_validation,
        "network": arguments.network,
        "use_cuda": use_cuda,
        "loss_function": loss_function,
        "tmp_path": arguments.tmp_path,
    }
    if arguments.network == 1:
        tuning_config["normalization"] = tune.choice(
            [nn.BatchNorm1d, nn.LazyBatchNorm1d, nn.SyncBatchNorm]
        )
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(training_loop),
            resources={"cpu": 1, "gpu": 0.25 if arguments.gpu > 0 else 0},
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler, num_samples=arguments.num_samples
        ),
        run_config=train.RunConfig(
            storage_path=arguments.path,
            log_to_file=True,
            checkpoint_config=train.CheckpointConfig(num_to_keep=5),
        ),
        param_space=tuning_config,
    )
    result = tuner.fit()
    best_trial = result.get_best_result("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.metrics['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.metrics['accuracy']}")

    # load best model
    checkpoint_path = os.path.join(
        best_trial.checkpoint.to_directory(), "ghost_checkpoint.pt"
    )
    model_state, _ = torch.load(checkpoint_path)

    # device for best model
    if use_cuda:
        device = torch.device(f"cuda:0")
    else:
        device = torch.device("cpu")

    if arguments.network == 0:
        model = GhostNetwork(
            num_features,
            l0=best_trial.config["l0"],
            activation=best_trial.config["activation"],
        )
    elif arguments.network == 1:
        model = GhostNetworkWithNormalization(
            num_features,
            l0=best_trial.config["l0"],
            activation=best_trial.config["activation"],
            normalization=best_trial.config["normalization"],
        )
    elif arguments.network == 2:
        model = GhostNetworkWithManualNormalization(
            num_features,
            l0=best_trial.config["l0"],
            matching=True,
            activation=best_trial.config["activation"],
            device=device,
        )
    model.load_state_dict(model_state)
    model = model.to(device)
    print()
    print(model)
    print(
        f"Model parameters: {sum([x.reshape(-1).shape[0] for x in model.parameters()])}"
    )
    print()
    # test accuracy
    test_dataset = GhostDataset(
        torch.tensor(data_test, dtype=torch.float32, device=device),
        torch.tensor(labels_test, dtype=torch.float32, device=device),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=arguments.batch)
    accuracy, loss = testing_loop(
        device, model, test_dataloader, loss_function, arguments.threshold
    )
    print(f"Test Accuracy: {accuracy * 100.0:.2f}%")
    print(f"Test Loss: {loss:.6f}")
    print()
    # save model
    if arguments.save:
        print("Saving model to disk")
        model = model.to("cpu")
        if arguments.network == 2:
            model.normalization.shift = model.normalization.shift.to("cpu")
            model.normalization.scale = model.normalization.scale.to("cpu")
        torch.save(model.state_dict(), "ghost_model.pth")
        with open("ghost_model_config.pkl", "wb") as file:
            pickle.dump(best_trial.config, file)
        print("Saving model to ONNX format")
        dummy_input = torch.randn(arguments.batch, num_features)
        dummy_input = dummy_input.to("cpu")
        torch.onnx.export(
            model,
            dummy_input,
            "ghost_model.onnx",
            input_names=["features"],
            output_names=["probabilities"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "probabilities": {0: "batch_size"},
            },
        )
    # INT8 quantization
    if arguments.int8:
        print("INT8 quantization")
        model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        model_prepared = torch.quantization.prepare_qat(model.train())
        for _ in range(0, num_epochs):
            inner_training_loop(
                model_prepared,
                DataLoader(training_dataset, batch_size=arguments.batch),
                "cpu",
                select_optimizer(best_trial.config, model),
                loss_function,
            )
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
            dummy_input = dummy_input.to("cpu")
            model_int8 = model_int8.to("cpu")
            torch.onnx.export(
                model_int8, dummy_input, "ghost_model_int8.onnx", export_params=True
            )


if __name__ == "__main__":
    __main__()
