import argparse
import os
from functools import partial
import logging
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import torch
import xgboost as xgboost
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch import nn
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from classification_performance_metrics import print_performance_metrics, save_performance_metrics
from utilities import (
    GhostDataset,
    QuietReporter,
    training_loop,
    inner_training_loop,
    testing_loop,
    select_optimizer,
)
from networks import GhostNetwork, GhostNetworkExperiment

CLASS_NAMES = ['real', 'ghost']


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
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=256)
    parser.add_argument(
        "-n",
        "--num_samples",
        help="Samples for hyperparameter tuning.",
        type=int,
        default=128,
    )
    # misc
    parser.add_argument("--threshold", help="Ghost threshold.", type=float, default=0.5)
    parser.add_argument(
        "--cpu", help="Number of CPU cores to use for training.", type=int, default=1
    )
    parser.add_argument(
        "--gpu", help="Number of GPUs to use for training.", type=int, default=0
    )
    parser.add_argument(
        "--int8", help="Quantize the trained model to INT8.", action="store_true"
    )
    parser.add_argument(
        "--save", help="Save the trained model to disk", action="store_true"
    )
    parser.add_argument('--learner_type', type=str,
                        help=f'ml model type; choose from ["nn", "xgboost","pcaxgboost", "rf"].', default='nn')
    return parser.parse_args()


def __main__():
    arguments = command_line()
    if not arguments.nocuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    # initialize ray
    ray.init(
        num_cpus=arguments.cpu,
        num_gpus=arguments.gpu,
        configure_logging=False,
        log_to_driver=False,
        logging_level=logging.ERROR,
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

    if arguments.learner_type == 'nn':
        train_and_tune_hyperparameters_and_evaluate_nn(training_dataset, validation_dataset, test_dataset,
                                                       arguments.save,
                                                       arguments.epochs, arguments.threshold, arguments.num_samples,
                                                       device,
                                                       arguments.cpu, arguments.gpu, arguments.int8)
    else:
        if arguments.learner_type == 'xgboost':
            predicted, probs, _feature_importances_ = train_and_tune_hyperparameters_and_evaluate_xgboost(
                training_dataset.data, test_dataset.data, training_dataset.labels, test_dataset.labels)
        elif arguments.learner_type == 'rf':
            predicted, probs, _feature_importances_ = train_and_tune_hyperparameters_and_evaluate_rf(
                training_dataset.data, test_dataset.data, training_dataset.labels, test_dataset.labels)
        elif arguments.learner_type == 'pcaxgboost':
            predicted, probs, _feature_importances_ = train_and_tune_hyperparameters_and_evaluate_pca_xgboost(
                training_dataset.data, test_dataset.data, training_dataset.labels, test_dataset.labels)
        probs = list(probs)
        test_labels = [class_id_to_label(class_id) for class_id in test_dataset.labels.numpy()[:, 0].astype(int)]
        output_path = Path(f'output_{arguments.learner_type}')
        os.makedirs(output_path, exist_ok=True)
        save_performance_metrics(test_labels, predicted, probs, CLASS_NAMES, output_path, positive_label=CLASS_NAMES[1])


def train_and_tune_hyperparameters_and_evaluate_pca_xgboost(X_train, X_validation, y_train, y_val):
    pca = PCA()
    pca = pca.fit(X_train)
    top_n_components = X_train.shape[1]
    X_train_pc = pca.transform(X_train)[:, :top_n_components]
    X_validation_pc = pca.transform(X_validation)[:, :top_n_components]

    return train_and_tune_hyperparameters_and_evaluate_xgboost(X_train_pc, X_validation_pc, y_train, y_val)


def class_id_to_label(class_id):
    return CLASS_NAMES[class_id]


def train_and_tune_hyperparameters_and_evaluate_xgboost(X_train, X_validation, y_train, y_val):
    n_trees = 500
    clf = xgboost.XGBClassifier(n_estimators=n_trees)

    clf.fit(X_train, y_train)

    predicted = [class_id_to_label(class_id) for class_id in clf.predict(X_validation)]
    probs = clf.predict_proba(X_validation)[:, 1]

    return predicted, probs, clf.feature_importances_


def train_and_tune_hyperparameters_and_evaluate_rf(X_train, X_validation, y_train, y_val):
    n_trees = 500
    clf = RandomForestClassifier(n_estimators=n_trees)

    clf.fit(X_train, y_train)

    predicted = [class_id_to_label(int(class_id)) for class_id in clf.predict(X_validation)]
    probs = clf.predict_proba(X_validation)[:, 1]

    return predicted, probs, clf.feature_importances_


def train_and_tune_hyperparameters_and_evaluate_nn(training_dataset, validation_dataset, test_dataset, save, epochs,
                                                   threshold, num_samples, device, cpu, gpu, int8):
    # training and tuning hyperparameters
    num_features = training_dataset.data.shape[1]
    num_epochs = epochs
    tuning_config = {
        "l0": tune.choice(
            [i for i in range(int(num_features / 3), int(num_features * 3), 4)]
        ),
        "learning": tune.loguniform(1e-6, 1e-1),
        "batch": tune.choice([2 ** i for i in range(1, 15)]),
        "epochs": tune.choice([num_epochs]),
        "optimizer": tune.choice([0, 1]),
        "activation": tune.choice(
            [
                nn.ReLU,
                nn.Tanh,
                nn.Sigmoid,
                nn.LeakyReLU,
                nn.ELU,
                nn.LogSigmoid,
                nn.Softmax,
                nn.Softmin,
            ]
        ),
        "normalization": tune.choice(
            [
                nn.BatchNorm1d,
                nn.LazyBatchNorm1d,
                nn.SyncBatchNorm,
                nn.InstanceNorm1d,
                nn.LayerNorm,
            ]
        ),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=num_epochs,
        grace_period=2,
        reduction_factor=2,
    )
    reporter = QuietReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    loss_function = nn.BCELoss()
    result = tune.run(
        partial(
            training_loop,
            num_features=num_features,
            device=device,
            loss_function=loss_function,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            threshold=threshold,
        ),
        resources_per_trial={"cpu": cpu, "gpu": gpu},
        config=tuning_config,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path="./ray_logs",
        checkpoint_score_attr="loss",
        progress_reporter=reporter,
        verbose=1,
    )
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    # load best model
    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    model = GhostNetwork(
        num_features,
        l0=best_trial.config["l0"],
        activation=best_trial.config["activation"],
        normalization=best_trial.config["normalization"],
    )
    model.load_state_dict(best_checkpoint.to_dict()["net_state_dict"])
    model.to(device)
    print()
    print(model)
    print(
        f"Model parameters: {sum([x.reshape(-1).shape[0] for x in model.parameters()])}"
    )
    print()
    # test accuracy
    test_dataloader = DataLoader(test_dataset, batch_size=best_trial.config["batch"])
    accuracy, loss = testing_loop(
        device, model, test_dataloader, loss_function, threshold
    )
    print(f"Test Accuracy: {accuracy * 100.0:.2f}%")
    print(f"Test Loss: {loss:.6f}")
    print()
    # save model
    if save:
        print("Saving model to disk")
        torch.save(model.state_dict(), "ghost_model.pth")
        with open("ghost_model_config.pkl", "wb") as file:
            pickle.dump(best_trial.config, file)
        print("Saving model to ONNX format")
        dummy_input = torch.randn(1, num_features)
        dummy_input.to("cpu")
        model.to("cpu")
        torch.onnx.export(model, dummy_input, "ghost_model.onnx", export_params=True)
    # INT8 quantization
    if int8:
        print("INT8 quantization")
        model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        model_prepared = torch.quantization.prepare_qat(model.train())
        for _ in range(0, num_epochs):
            inner_training_loop(
                model_prepared,
                DataLoader(
                    training_dataset, batch_size=int(best_trial.config["batch"])
                ),
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
        if save:
            print("Saving INT8 model to disk")
            torch.save(model_int8, "ghost_model_int8.pth")
            print("Saving INT8 model to ONNX format")
            dummy_input = torch.randn(1, num_features)
            dummy_input.to("cpu")
            model_int8.to("cpu")
            torch.onnx.export(
                model_int8, dummy_input, "ghost_model_int8.onnx", export_params=True
            )


if __name__ == "__main__":
    __main__()
