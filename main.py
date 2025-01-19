"""
CLI entrypoint for wildfire prediction.
"""

import click
import torch
from torch.utils.data import DataLoader

from wildfire_prediction.dataset import WildfireDataset
from wildfire_prediction.models.resnext import ResnextClassifier
from wildfire_prediction.test.classifier import test_classifier
from wildfire_prediction.utils.cli import (
    batch_size,
    checkpoints,
    classifier,
    device,
    save_results,
)


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


if __name__ == "__main__":
    main()
