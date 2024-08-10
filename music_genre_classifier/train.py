import argparse
import torch
from torch import nn
from torchsummary import summary
import torch.optim as optim
from model import CNNNetwork
from data_managment import dataset
import numpy as np
import sklearn.metrics
import os
import mlflow
from tqdm import tqdm

mlflow.set_experiment("music-genre-classifier")


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.dataloader.DataLoader,
    valid_dataloader: torch.utils.data.dataloader.DataLoader,
    epochs: int,
    loss_function: torch.nn.modules.loss,
    optimiser: torch.optim,
    device: str,
) -> tuple:
    """Trains and validates a CNN model.

    Parameters
    ----------
    model : torch.nn.Module
        CNN model to be trained.
    train_dataloader : torch.utils.data.DataLoader
        Training dataloader.
    valid_dataloader : torch.utils.data.DataLoader
        Validation dataloader.
    test_dataloader : torch.utils.data.DataLoader
        Testing dataloader.
    epochs : int > 0
        Number of epochs.
    loss_function : torch.nn.modules.loss
        Loss function.
    optimiser : torch.optim
        Optimiser.
    device : str
        Device to process. CPU or GPU (cuda).

    Returns
    -------
    model : torch.nn.Module
        Model trained to be saved.
    history : dict
        Dictionary with training and validation information (loss and accuracy).
    """
    history = {
        "training_loss": [],
        "training_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    for epoch in range(epochs):

        # Training
        model, train_loss, train_accuracy = train_single_epoch(
            model, train_dataloader, loss_function, optimiser, device
        )
        history["training_loss"].append(train_loss)
        history["training_accuracy"].append(train_accuracy)

        # Validation
        val_loss, val_accuracy = validate_single_epoch(
            model, valid_dataloader, loss_function, device
        )
        history["validation_loss"].append(val_loss)
        history["validation_accuracy"].append(val_accuracy)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        print(
            f"Epoch {epoch+1} | {epochs} --> Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.4f}"
        )
        print(
            f"Epoch {epoch+1} | {epochs} --> Validation loss: {val_loss:.4f} | Validation accuracy: {val_accuracy:.4f}"
        )
        print(
            "-------------------------------------------------------------------------"
        )

    return model, history


def train_single_epoch(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.dataloader.DataLoader,
    loss_function: torch.nn.modules.loss,
    optimiser: torch.optim,
    device: str,
) -> tuple:
    """Trains a single epoch of a model's training.

    Parameters
    ----------
    model : torch.nn.Module
        CNN model to be trained.
    train_dataloader : torch.utils.data.DataLoader
        Training dataloader.
    loss_function : torch.nn.modules.loss
        Loss function.
    optimiser : torch.optim
        Optimiser.
    device : str
        Device to process. CPU or GPU (cuda).

    Returns
    -------
    model : torch.nn.Module
        Model trained in a single epoch.
    train_loss: float
        Training loss in a single epoch.
    train_accuracy: float
        Training accuracy in a single epoch.
    """
    train_loss = []
    y_true = []
    y_pred = []
    model.train()
    for data, labels in tqdm(train_dataloader, desc="Training", unit="batch"):
        data, labels = data.to(device), labels.to(device)

        prediction = model(data)  # Predict data with the model
        loss = loss_function(
            prediction, labels
        )  # Loss between predicted values and label values
        optimiser.zero_grad()  # Zero out the gradients
        loss.backward()  # Backpropagation. Computes gradients
        optimiser.step()  # Updates the weights
        train_loss.append(loss.item())

        # Accuracy
        _, pred = torch.max(prediction.data, 1)  # Prediction with most probability
        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist())

    train_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    train_loss = np.mean(train_loss)

    return model, train_loss, train_accuracy


def validate_single_epoch(
    model: torch.nn.Module,
    valid_dataloader: torch.utils.data.dataloader.DataLoader,
    loss_function: torch.nn.modules.loss,
    device: str,
) -> tuple:
    """Validates a single epoch of a model's training.

    Parameters
    ----------
    model : torch.nn.Module
        CNN model to be trained.
    valid_dataloader : torch.utils.data.DataLoader
        Validation dataloader.
    loss_function : torch.nn.modules.loss
        Loss function.
    device : str
        Device to process. CPU or GPU (cuda).

    Returns
    -------
    valid_loss: float
        Validation loss in a single epoch.
    val_accuracy: float
        Validation accuracy in a single epoch.
    """
    valid_loss = []
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for data, labels in tqdm(valid_dataloader, desc="Validation", unit="batch"):
            data, labels = data.to(device), labels.to(device)

            prediction = model(data)  # Predict data with the model
            loss = loss_function(
                prediction, labels
            )  # Loss between predicted values and label values
            valid_loss.append(loss.item())

            # Accuracy
            _, pred = torch.max(prediction.data, 1)  # Prediction with most probability
            y_true.extend(labels.tolist())
            y_pred.extend(pred.tolist())

    val_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    valid_loss = np.mean(valid_loss)

    return valid_loss, val_accuracy


def main(args):
    with mlflow.start_run():

        mlflow.set_tag("developer", "Franco Bach")

        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss_function", "CrossEntropyLoss")
        mlflow.log_param("dataset_path", args.dataset_path)

        # Dataset
        train_dataloader = dataset.get_train_loader(args.dataset_path)
        valid_dataloader = dataset.get_valid_loader(args.dataset_path)

        # Device
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Construct a model
        cnn = CNNNetwork().to(device)
        summary(cnn, (1, 128, 130))

        # Initialise Loss function and Optimizer
        loss_function = nn.CrossEntropyLoss()
        optimiser = optim.Adam(cnn.parameters(), lr=args.learning_rate)

        # Train model
        model, history = train(
            model=cnn,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            epochs=args.epochs,
            loss_function=loss_function,
            optimiser=optimiser,
            device=device,
        )

        # Save model
        if not os.path.exists("results"):
            os.makedirs("results")
        torch.save(model.state_dict(), "results/music_genre_classifier.pth")

        # Log model in MLFlow
        mlflow.pytorch.log_model(cnn, "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a music genre classification model."
    )
    parser.add_argument(
        "--dataset_path", type=str, default="dataset", help="Base path to the dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )

    args = parser.parse_args()
    main(args)
