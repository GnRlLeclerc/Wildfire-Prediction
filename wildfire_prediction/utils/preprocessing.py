import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

from wildfire_prediction.utils.files import recursive_count_files, recursive_list_files


def compute_image_stats(path: str, batch_size=256) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and std of each channel of the images in the given dataset directory
    for normalization.

    It is computed in a batched manner to avoid memory issues.
    """

    image_count = recursive_count_files(path)

    batch_count = (image_count + batch_size - 1) // batch_size

    # E(X) for later use as E(X)²
    means = np.zeros((batch_count, 3))
    # E(X²)
    square_means = np.zeros((batch_count, 3))

    i = 0
    batch_index = 0

    images = np.zeros((batch_size, 350, 350, 3))

    for path in tqdm(recursive_list_files(path), total=image_count):

        if i == batch_size:
            # Reset the batch
            i = 0
            batch_index += 1

            # Do the computation
            means[batch_index] = np.mean(images, axis=(0, 1, 2))
            square_means[batch_index] = np.mean(images**2, axis=(0, 1, 2))

        # Load the next image into the batch
        try:
            images[i] = np.array(Image.open(path))
        except:
            # If for some reason the image is corrupted, skip it
            continue

        i += 1

    # Flush the remaining computation
    if i > 0:
        means[batch_index] = np.mean(images, axis=(0, 1, 2))
        square_means[batch_index] = np.mean(images**2, axis=(0, 1, 2))

    actual_mean = means.mean(axis=0)
    actual_std = np.sqrt(square_means.mean(axis=0) - actual_mean**2)

    return actual_mean, actual_std


def compute_train_test_split(path: str, train_ratio=0.8, seed=42):
    """Compute a train / test split from the "valid" directory.
    ("train" is forbidden, "test" is reserved for the final test set.)

    Args:
        path (str): Path to the dataset directory. Do not include the "valid" directory in the path.
    """

    random.seed(seed)

    train: list[tuple[str, int]] = []
    test: list[tuple[str, int]] = []

    for path in recursive_list_files(os.path.join(path, "valid")):

        is_wildfire = int("nowildfire" not in path)
        sample = (path, is_wildfire)

        if random.random() < train_ratio:
            train.append(sample)
        else:
            test.append(sample)

    train_array = np.array(train)
    test_array = np.array(test)

    np.savetxt("train.csv", train_array, fmt="%s", delimiter=";")
    np.savetxt("test.csv", test_array, fmt="%s", delimiter=";")


def get_unlabeled_train(path: str, ratio: float = 0.5, seed: int = 42):
    """Get a train unlabeled data from the "train" directory.

    Args:
        path (str): Path to the dataset directory. Do not include the "train" directory in the path.
        ratio (float): Defines how much train data to use.
    """

    random.seed(seed)

    train: list[tuple[str, None]] = []

    for path in recursive_list_files(os.path.join(path, "train")):
        sample = (path, None)
        if random.random() < ratio:
            train.append(sample)

    train_array = np.array(train)

    np.savetxt("train_unlabeled.csv", train_array, fmt="%s", delimiter=";")
