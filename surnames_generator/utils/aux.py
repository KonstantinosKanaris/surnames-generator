"""
Contains various utility functions for PyTorch model training.
"""

from __future__ import annotations

import errno
import os
import time
from datetime import timedelta
from typing import Any, Dict

import torch
import yaml

from surnames_generator import logger


def load_yaml_file(filepath: str) -> Any:
    """Loads a `yaml` configuration file into a dictionary.

    Args:
        filepath (str): The path to the `yaml` file.

    Returns:
        Any: The configuration parameters.
    """
    if not os.path.isfile(path=filepath):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

    with open(file=filepath, mode="r", encoding="utf-8") as f:
        config = yaml.full_load(stream=f)
    logger.info("Configuration file loaded successfully.\n")

    return config


def set_seeds(seed: int = 42) -> None:
    """Sets random seeds for torch operations.

    Args:
      seed (int, optional): Random seed to set (default=42).
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)

    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def load_general_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str
) -> Dict[str, Any]:
    """Loads a general checkpoint.

    Args:
        model (torch.nn.Module):
            The model to be updated with its saved `state_dict`.
        optimizer (torch.optim.Optimizer):
            The optimizer to be updated with its saved `state_dict`.
        filepath (str): The file path of the general checkpoint.

    Returns:
        A dictionary containing the following keys:
            - 'model': The updated model with its saved `state_dict`.
            - 'optimizer': The updated optimizer with its saved `state_dict`.
            - 'epoch': The epoch value from the last checkpoint.
            - 'loss': The loss value from the last checkpoint.
    """
    checkpoint = torch.load(f=filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "model": model,
        "optimizer": optimizer,
        "epoch": checkpoint["epoch"],
        "val_loss": checkpoint["loss"],
    }


class Timer:
    """Context manager to count elapsed time.

    Example:

        >>> def do_something():
        ...     pass
        >>>
        >>> with Timer() as t:
        ...   do_something()
        >>> print(f"Invocation of f took {t.elapsed}s!")
    """

    def __enter__(self) -> Timer:
        """
        Starts the time counting.

        Returns:
          Timer: An instance of the `Timer` class.
        """
        self._start = time.time()
        return self

    def __exit__(self, *args: int | str) -> None:
        """
        Stops the time counting.

        Args:
          args (int | str)
        """
        self._end = time.time()
        self._elapsed = self._end - self._start
        self.elapsed = str(timedelta(seconds=self._elapsed))
