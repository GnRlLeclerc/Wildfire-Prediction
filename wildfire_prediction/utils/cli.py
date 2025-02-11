"""Reusable CLI flags and options"""

import click
import torch


def batch_size(func):
    return click.option(
        "--batch-size",
        default=32,
        help="The batch size to use for training/testing",
        type=int,
    )(func)


def device(func):
    return click.option(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=click.Choice(["cpu", "cuda"]),
        help="The device to use. Defaults to 'cuda' if available, otherwise 'cpu'",
    )(func)


def classifier(func):
    return click.option(
        "--classifier",
        help="The classifier to use",
        type=click.Choice(
            ["resnext", "vit_b_16", "vit_b_32", "alexnet", "mean_teacher"]
        ),
    )(func)


def checkpoints(func, required=True):
    return click.option(
        "--checkpoints",
        help="The path to the model checkpoints",
        required=required,
        type=str,
    )(func)


def save_results(func):
    return click.option(
        "--save-results",
        default=None,
        help="The path to save the results. If not provided, results will not be saved.",
        type=str,
    )(func)
