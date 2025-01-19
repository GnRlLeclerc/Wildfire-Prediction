"""
CLI entrypoint for wildfire prediction.
"""

import click
import torch
from torch.utils.data import DataLoader

from wildfire_prediction.dataset import WildfireDataset
from wildfire_prediction.models.resnext import ResnextClassifier
from wildfire_prediction.test.classifier import test_classifier
from wildfire_prediction.train.classifier import train_classifier
from wildfire_prediction.utils.cli import (
    batch_size,
    checkpoints,
    classifier,
    device,
    save_results,
)
from wildfire_prediction.utils.results import Results


@click.group()
def main():
    pass


@main.command()
@classifier
@batch_size
@checkpoints
@save_results
@device
def test(
    classifier: str,
    checkpoints: str,
    batch_size: int,
    device: str,
    save_results: str | None,
):
    """Test classifiers."""

    match classifier:
        case "resnext":
            model = ResnextClassifier()
        case _:
            raise ValueError(f"Unknown classifier variant: {classifier}")

    # Load the model checkpoints
    model.load_state_dict(torch.load(checkpoints, weights_only=True))

    # Load the dataset
    dataset = WildfireDataset("test")
    dataloader = DataLoader(dataset, batch_size=batch_size)

    results = test_classifier(model, dataloader, device)

    print(results)

    if save_results is not None:
        with open(save_results, "w") as f:
            f.write(results.to_json())


@main.command()
@classifier
@batch_size
@device
@click.option(
    "--epochs",
    help="The amount of epochs to train the model for",
    type=int,
)
@click.option(
    "--learning-rate",
    help="The optimizer learning rate",
    type=float,
)
def train(
    classifier: str,
    batch_size: int,
    device: str,
    epochs: int,
    learning_rate: float,
):
    """Train classifiers."""

    match classifier:
        case "resnext":
            model = ResnextClassifier()
        case _:
            raise ValueError(f"Unknown classifier variant: {classifier}")

    # Load the dataset
    train_dataset = WildfireDataset("train")
    test_dataset = WildfireDataset("test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    train_classifier(model, train_loader, test_loader, epochs, learning_rate, device)


@main.command()
@click.option(
    "--path",
    help="The path to the logs",
    type=str,
)
def logs(path: str):
    """Plot training logs."""

    training: list[Results] = []
    testing: list[Results] = []

    # Read the log file
    with open(path, "r") as f:
        for line in f:
            result = Results.from_json(line)
            if result.mode == "train":
                training.append(result)
            else:
                testing.append(result)

    Results.plot_loss(training)
    Results.plot_metrics(training, testing)


if __name__ == "__main__":
    main()
