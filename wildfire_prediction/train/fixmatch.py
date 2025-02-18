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
from wildfire_prediction.utils.images import CutoutDefault, RandAugment

transform_weak = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode = 'reflect'),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    #CutoutDefault(cutout),
    ])#"weak" data augmentation

transform_strong = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode = 'reflect'),

    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    RandAugment(N=2, M=10),  # N=2: number of augmentations, M=5: magnitude
    CutoutDefault(16),
    # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),

    ]) # strong data transformation

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
    """Train mean teacher"""
    model.to(device)
    optimizer = Adam(model.student.parameters(), lr=learning_rate)
    log_file = "training_logs.jsonl"

    for i in tqdm(range(epochs), desc="Training Classifier with Fixmatch"):

        model.train()
        results = Results(iteration=i, mode="train", loss=0)

        for (labeled_data, labels), unlabeled_data in zip(
            train_loader_labeled, train_loader_unlabeled
        ):
                    
            labeled_data = transform_weak(labeled_data)
            unlabeled_data_weak = transform_weak(unlabeled_data)
            unlabeled_data_strong = transform_strong(unlabeled_data)

            labeled_data, labels = labeled_data.to(device), labels.to(device)
            unlabeled_data_weak, unlabeled_data_strong = unlabeled_data_weak.to(device), unlabeled_data_strong.to(device)

            inputs_full = torch.cat([labeled_data, unlabeled_data_weak, unlabeled_data_strong], dim=0)

            outputs_full = model(inputs_full)

            outputs_sup = outputs_full[0:len(labeled_data)]
            outputs_strong_weak = outputs_full[len(labeled_data)::]
            outputs_weak, outputs_strong = torch.chunk(outputs_strong_weak, 2)

            weak_logits = F.softmax(outputs_weak, dim=-1)
            confidences, pseudo_labels = torch.max(weak_logits, dim=1)
            confident_mask = confidences > threshold

                  # Partie pour verifier l'accuracy des pseudo labels
            total2 += confident_mask.sum().item()
            correct2 += (pseudo_labels[confident_mask] == labels[confident_mask]).sum().item()

            # Calcul des loss et
            unlab_loss = F.cross_entropy(outputs_strong[confident_mask], pseudo_labels[confident_mask])
            lab_loss = F.cross_entropy(outputs_sup, labels)
            total_loss = unlab_loss + lab_loss

            total_loss.backward()  # Backward Propagation
            optimizer.step() # Optimizer update
            unlab_loss = 0
            results.loss += total_loss.item()

            _, predicted = torch.max(outputs_sup.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()

            del outputs_full
            torch.cuda.empty_cache()

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