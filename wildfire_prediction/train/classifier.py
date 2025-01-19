"""Train classifiers"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from wildfire_prediction.models.base import Classifier
from wildfire_prediction.test.classifier import test_classifier
from wildfire_prediction.utils.results import Results


def train_classifier(
    classifier: Classifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
):
    """Train a classifier"""

    classifier.to(device)
    optimizer = Adam(classifier.parameters(), lr=learning_rate)
    log_file = "training_logs.jsonc"

    for i in tqdm(range(epochs), desc="Training classifier"):

        classifier.train()
        results = Results(iteration=i, mode="train", loss=0)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = classifier(images).squeeze()
            results.add_predictions(outputs, labels)

            # Compute the loss
            loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
            loss.backward()

            # Update the weights
            optimizer.step()

        # Append training logs
        results.compute_metrics()
        results.append_log(log_file)

        results = test_classifier(classifier, test_loader, device, verbose=False)
        results.mode = "test"
        results.iteration = i
        results.append_log(log_file)

        # Save the model checkpoints every 20% of the epochs
        if i % (epochs // 5) == 0:
            torch.save(classifier.state_dict(), f"classifier-{i}-{epochs}.pth")

    # Save the final model
    torch.save(classifier.state_dict(), "classifier-final.pth")
