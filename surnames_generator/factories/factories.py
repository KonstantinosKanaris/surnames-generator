from typing import Dict, Iterator

import torch


class ModelsFactory:
    """A factory class for registering and instantiating
    models.

    Attributes:
        _models (Dict[str, Any]): A dictionary to store
            registered models.
    """

    def __init__(self) -> None:
        self._models: Dict[str, type[torch.nn.Module]] = {}

    def register_model(self, name: str, model: type[torch.nn.Module]) -> None:
        """Registers a PyTorch model with the given name.

        Args:
            name (str): The name of the model to register.
            model (type[torch.nn.Module]): The model's class
                type.
        """
        self._models[name] = model

    def get_model(self, name: str, **kwargs) -> torch.nn.Module:
        """Instantiates and returns a model by name.

        Args:
            name (str): The name of the model to instantiate.

        Returns:
            Any: An instance of the specified model.
        """
        model = self._models[name]
        return model(**kwargs)


class OptimizersFactory:
    """A factory class for registering and instantiating
    optimizers.

    Attributes:
        _optimizers (Dict[str, type[torch.optim.Optimizer]]):
            A dictionary to store registered optimizers.
    """

    def __init__(self) -> None:
        self._optimizers: Dict[str, type[torch.optim.Optimizer]] = {}

    def register_optimizer(
        self, name: str, optimizer: type[torch.optim.Optimizer]
    ) -> None:
        """
        Registers a PyTorch optimizer with the given name.

        Args:
            name (str): The name of the optimizer to register.
            optimizer (type[torch.optim.Optimizer]):
                The optimizer's class type.
        """
        self._optimizers[name] = optimizer

    def get_optimizer(
        self, name: str, model_params: Iterator[torch.nn.parameter.Parameter], **kwargs
    ) -> torch.optim.Optimizer:
        """Instantiates and returns a PyTorch optimizer by name.

        Args:
            name (str): The name of the optimizer to instantiate.
            model_params (Iterator[torch.nn.parameter.Parameter]):
                Iterator of the model parameters.

        Returns:
            torch.optim.Optimizer: An instance of the specified
                optimizer.
        """
        optimizer = self._optimizers[name]
        return optimizer(model_params, **kwargs)


class LossFactory:
    """
    A factory class for registering and instantiating loss classes.

    Attributes:
        _losses (Dict[str, type[torch.nn.Module]]):
            A dictionary to store registered loss classes.
    """

    def __init__(self) -> None:
        self._losses: Dict[str, type[torch.nn.Module]] = {}

    def register_loss(self, name: str, loss: type[torch.nn.Module]) -> None:
        """
        Registers a loss class with the given name.

        Args:
            name (str): The name of the loss class to register.
            loss (type[torch.nn.Module]): The loss class type.
        """
        self._losses[name] = loss

    def get_loss(self, name: str, **kwargs) -> torch.nn.Module:
        """Returns an instance of the registered loss class based on
        the input name.

        Args:
            name (str): The name of the loss class to instantiate.

        Returns:
            torch.nn.Module: An instance of the specified loss class.
        """
        loss = self._losses[name]
        return loss(**kwargs)
