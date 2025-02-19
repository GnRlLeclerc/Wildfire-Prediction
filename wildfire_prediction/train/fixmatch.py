"""Train classifiers"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from wildfire_prediction.models.base import Classifier
from wildfire_prediction.test.classifier import test_classifier
from wildfire_prediction.utils.results import Results
from wildfire_prediction.utils.images import CutoutDefault, RandAugment, from_torch

transform_weak = transforms.Compose([
    transforms.RandomCrop(size=(224, 224), padding=28, padding_mode = 'reflect'),

    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    #CutoutDefault(cutout),
    ])#"weak" data augmentation

transform_strong = transforms.Compose([
    # transforms.toPILImage(),
    transforms.RandomCrop(size=(224, 224), padding=28, padding_mode = 'reflect'), # shift image up to 12.5% of the image size

    transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    RandAugment(N=2, M=10),  # N=2: number of augmentations, M=5: magnitude
    CutoutDefault((224//2)),
    # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),

    ]) # strong data transformation

def plot_images(images, n_image, savefig=False, filename="images.png"):    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots((n_image + 9) // 10, min(n_image, 10), figsize=(20, (n_image + 9) // 10 * 2))
    fig.patch.set_facecolor('black')
    axs = axs.flatten()
    for i, image in enumerate(images):
        axs[i].imshow(from_torch(image))
        axs[i].axis('off')
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    plt.tight_layout()
    if savefig:
        plt.savefig(filename)
    plt.close()


def train_FixMatch_classifier(
    model: Classifier,
    train_loader_labeled: DataLoader,
    train_loader_unlabeled: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    threshold: float,
    device: str,
):
    """Train Classifier with FixMatch."""
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0008)
    log_file = "training_logs.jsonl"
    for i in tqdm(range(epochs), desc="Training Classifier with Fixmatch"):

        model.train()
        results = Results(iteration=i, mode="train", loss=0)

        for (labeled_data, labels), unlabeled_data in zip(
            train_loader_labeled, train_loader_unlabeled
        ):
            optimizer.zero_grad()

            # Data Augmentation
            labeled_data = transform_weak(labeled_data)
            unlabeled_data_weak = transform_weak(unlabeled_data)
            unlabeled_data_strong = transform_strong(unlabeled_data)

            labeled_data, labels = labeled_data.to(device), labels.to(device)
            unlabeled_data_weak, unlabeled_data_strong = unlabeled_data_weak.to(device), unlabeled_data_strong.to(device)
            labels = labels.float()

            inputs_full = torch.cat([labeled_data, unlabeled_data_weak, unlabeled_data_strong], dim=0)

            outputs_full = model(inputs_full)

            outputs_sup = outputs_full[0:len(labeled_data)]
            outputs_strong_weak = outputs_full[len(labeled_data)::]
            outputs_weak, outputs_strong = torch.chunk(outputs_strong_weak, 2)

            # Pseudo-labeling
            weak_logits = F.sigmoid(outputs_weak)
            pseudo_labels = (weak_logits > 0.5).float()
            confident_mask = (weak_logits > threshold) | (weak_logits < (1 - threshold))

            # Calcul des loss
            unlab_loss = F.binary_cross_entropy_with_logits(outputs_strong[confident_mask], pseudo_labels[confident_mask])
            lab_loss = F.binary_cross_entropy_with_logits(outputs_sup, labels.unsqueeze(1))
            total_loss = unlab_loss + lab_loss

            total_loss.backward()  # Backward Propagation
            optimizer.step() # Optimizer update
            unlab_loss = 0
            results.loss += total_loss.item()
            results.add_predictions(outputs_sup.squeeze(1), labels.int())

            del outputs_full
            # torch.cuda.empty_cache()

        # Append training logs
        results.compute_metrics()
        results.append_log(log_file)

        results = test_classifier(model, test_loader, device, verbose=False)
        results.mode = "test"
        results.iteration = i
        results.append_log(log_file)

        # Save the model checkpoints every 20% of the epochs
        checkpoint_frequency = max(epochs // 5, 1)
        if i % checkpoint_frequency == 0:
            torch.save(model.state_dict(), f"fixmatch-{type(model).__name__}-{i}-{epochs}.pth")

    # Save the final model
    torch.save(model.state_dict(), f"fixmatch-{type(model).__name__}-final.pth")