from typing import Any, Dict, Iterator

import torch

from surnames_generator import logger
from surnames_generator.factories.factories import (
    LossFactory,
    ModelsFactory,
    OptimizersFactory,
)
from surnames_generator.models import GRUGenerator, LSTMGenerator, SimpleRNNGenerator
from surnames_generator.utils.vectorizers import CharacterVectorizer


class Client:
    """Provide methods to select different models, optimizers, and
    loss functions.

    Internally uses factories to manage the creation (initialization)
    of these components.

    Args:
        init_params (Dict[str, Any]): A set of parameters for the
            initialization of the model, optimizer, and loss.
        vectorizer (CharacterVectorizer): The dataset's vectorizer.

    Attributes:
        init_params (Dict[str, Any]): A set of parameters for the
            initialization of the model, optimizer, and loss.
        vectorizer (CharacterVectorizer): The dataset's vectorizer.
        models_factory (ModelsFactory): Factory for creating models.
        optimizers_factory (OptimizersFactory): Factory for creating
            optimizers.
        loss_factory (LossFactory): Factory for creating loss functions.
    """

    _available_models = ["simple_rnn_generator", "gru_generator", "lstm_generator"]
    _available_optimizers = ["adam", "sgd"]
    _available_losses = ["nll", "cross_entropy"]

    def __init__(
        self, init_params: Dict[str, Any], vectorizer: CharacterVectorizer
    ) -> None:
        self.init_params: Dict[str, Any] = init_params
        self.vectorizer: CharacterVectorizer = vectorizer

        self.models_factory: ModelsFactory = ModelsFactory()
        self.optimizers_factory: OptimizersFactory = OptimizersFactory()
        self.loss_factory: LossFactory = LossFactory()

    def model_client(self, name: str) -> torch.nn.Module:
        """Selects and returns a model instance based on the
        provided model name.

        The model name and the initialization parameters are
        defined in the configuration file. The available models
        are: ``'simple_rnn_generator'``, ``'gru_generator'`` and
        ``'lstm_generator'``.

        Args:
            name (str): The name of the model.

        Returns:
            torch.nn.Module: An instance of the selected model.

        Raises:
            ValueError: If the provided model name doesn't exist.
        """
        model_kwargs = {
            "char_vocab_size": len(self.vectorizer.char_vocab),
            "num_categories": len(self.vectorizer.category_vocab),
            "padding_idx": self.vectorizer.char_vocab.mask_index,
            "batch_first": True,
        }
        match name.lower():
            case "simple_rnn_generator":
                self.models_factory.register_model(
                    name=name.lower(), model=SimpleRNNGenerator
                )
            case "gru_generator":
                self.models_factory.register_model(
                    name=name.lower(),
                    model=GRUGenerator,
                )
            case "lstm_generator":
                self.models_factory.register_model(
                    name=name.lower(), model=LSTMGenerator
                )
            case _:
                raise ValueError(
                    f"Unsupported model name: `{name}`. "
                    f"Choose one of the following: {self.__class__._available_models}"
                )

        model_kwargs.update(self.init_params["model_init_params"])

        model = self.models_factory.get_model(name=name.lower(), **model_kwargs)
        return model

    def optimizer_client(
        self, name: str, model_params: Iterator[torch.nn.parameter.Parameter]
    ) -> torch.optim.Optimizer:
        """Selects and returns an optimizer instance based
        on the provided optimizer name.

        The optimizer name and the initialization parameters are
        defined in the configuration file. The available optimizers  are:
        ``'sgd'`` and  ``'adam'``.

        Args:
            name (str): The name of the optimizer.
            model_params (Iterator[torch.nn.parameter.Parameter]):
                The model's parameters.

        Returns:
            torch.optim.Optimizer: An instance of the selected
                optimizer.

        Raises:
            ValueError: If the provided optimizer name doesn't
                exist.
        """
        match name.lower():
            case "sgd":
                self.optimizers_factory.register_optimizer(
                    name=name.lower(), optimizer=torch.optim.SGD
                )
            case "adam":
                self.optimizers_factory.register_optimizer(
                    name=name.lower(),
                    optimizer=torch.optim.Adam,
                )
            case _:
                logger.error(f"Unsupported optimizer name: `{name}`.")
                raise ValueError(
                    f"Unsupported optimizer name: `{name}`. "
                    f"Choose one of the following: {self.__class__._available_optimizers}"
                )

        optimizer_kwargs = self.init_params["optimizer_init_params"]

        optimizer = self.optimizers_factory.get_optimizer(
            name=name.lower(), model_params=model_params, **optimizer_kwargs
        )
        return optimizer

    def loss_client(self, name: str) -> torch.nn.Module:
        """Selects and returns a loss instance based on the
        provided loss name.

        The loss name is defined in the configuration file and
        the available losses  are: ``'cross_entropy'`` and
        ``'nll'``.

        Note:
            If ``'nll'`` is selected, ``nn.LogSoftmax()`` is applied
            to the prediction logits.

        Args:
            name (str): The name of the loss class.

        Returns:
            torch.nn.Module: An instance of the selected loss.

        Raises:
            ValueError: If the provided loss name doesn't exist.
        """
        match name.lower():
            case "cross_entropy":
                self.loss_factory.register_loss(
                    name=name.lower(), loss=torch.nn.CrossEntropyLoss
                )
            case "nll":
                self.loss_factory.register_loss(
                    name=name.lower(),
                    loss=torch.nn.NLLLoss,
                )
            case _:
                logger.error(f"Unsupported loss name: `{name}`. ")
                raise ValueError(
                    f"Unsupported loss name: `{name}`."
                    f"Choose one of the following: {self.__class__._available_losses}"
                )

        loss_kwargs = {"ignore_index": self.vectorizer.char_vocab.mask_index}

        loss = self.loss_factory.get_loss(name=name.lower(), **loss_kwargs)
        return loss
