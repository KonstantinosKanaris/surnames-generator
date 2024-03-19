"""Supported functionality for the training process."""

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from surnames_generator import logger


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers
    in the net during training. Can be used for checking for
    possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the
    gradient flow.
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=1)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )


def normalize_tensors(
    y_pred: torch.Tensor, y_true: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalizes tensor sizes.

    If :attr:`y_pred` is a 3-D tensor, reshapes it to a 2-D matrix.
    If :attr:`y_true` is a 2-D matrix, reshapes it to a 1-D vector.

    Args:
        y_pred (torch.Tensor): The output of the model.
        y_true (torch.Tensor): The target predictions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The normalized tensors.
    """
    if y_pred.ndim == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if y_true.ndim == 2:
        y_true = y_true.contiguous().view(-1)

    return y_pred, y_true


class EarlyStopping:
    """Implements early stopping during training.

    Args:
        patience (int, optional):
            Number of epochs to wait before early stopping.
            (default=5).
        delta (float, optional):
            Minimum change in monitored quantity to qualify
            as an improvement (default=0).
        verbose (bool, optional):
            If ``True``, prints a message for each improvement.
            Defaults to `False`.
        path (str, optional):
            Path to save the checkpoint. Should include either
            `.pth` or `.pt` as the file extension. Defaults to
            ``'./checkpoint.pt'``.
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0,
        verbose: bool = False,
        path: str = "./checkpoint.pt",
    ) -> None:
        assert os.path.basename(path).endswith(
            (".pth", ".pt")
        ), "model_name should end with '.pt' or '.pth'"

        self.patience: int = patience
        self.delta: float = delta
        self.verbose: bool = verbose
        self.path: str = path

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min = np.Inf

    def __call__(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        val_loss: float,
    ) -> None:
        """Call method to check if the model's performance has
        improved.

        Args:
            epoch (int): Current epoch.
            model (torch.nn.Module): Model to be saved if the
                performance improves.
            optimizer (torch.optim.Optimizer): Optimizer used for
                training.
            val_loss (float): Validation loss to be monitored.
        """
        score = -val_loss

        if not self.best_score:
            self.best_score = score
            self.save_general_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
            )
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_general_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
            )
            self.counter = 0

    def save_general_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        val_loss: float,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Saves a general checkpoint during training.

        In addition to the model's `state_dict`, a general checkpoint
        also includes the optimizer's `state_dict`, the current epoch,
        and the validation loss value.

        Args:
            epoch (int): Current epoch.
            model (torch.nn.Module): Model to be saved.
            val_loss (float): Validation loss at the time of saving
                the checkpoint.
            optimizer (torch.optim.Optimizer): Optimizer used for
                training.
        """
        if not os.path.isdir(s=os.path.dirname(self.path)):
            os.makedirs(name=os.path.dirname(self.path), exist_ok=True)

        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> "
                f"{val_loss:.6f}). Saving model to {self.path}"
            )

        torch.save(
            obj={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            f=self.path,
        )
        self.val_loss_min = val_loss
