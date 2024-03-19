import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import mlflow
import pandas as pd  # noqa: F401
import torch.utils.data
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, SubsetRandomSampler

from surnames_generator import logger
from surnames_generator.datasets import SurnamesDataset
from surnames_generator.engine.generator import SurnamesGenerator
from surnames_generator.engine.trainer import Trainer
from surnames_generator.factories.client import Client
from surnames_generator.utils.aux import Timer, set_seeds
from surnames_generator.utils.vectorizers import CharacterVectorizer


class TrainingController:
    """Controls the flow of a PyTorch text generation training process.

    Handles the selection and configuration of the model, loss function,
    optimizer, and lr scheduler. Additionally, manages the creation of
    training and validation dataloaders and finally executes the training
    process.

    Args:
        dataset (SurnamesDataset): The surnames dataset for training.
        hyperparameters (Dict[str, Any]): Set of hyperparameters for
            the training process.
        checkpoints_dir (str): Directory to save checkpoints during
            training.
        resume (optional, bool): If ``True``, resumes training from
            a saved checkpoint. Defaults to ``False``.

    Attributes:
        dataset (SurnamesDataset): The surnames dataset for training.
        hyperparameters (Dict[str, Any]): Set of hyperparameters for
            the training process.
        checkpoints_dir (str): Directory to save checkpoints during
            training.
        resume (bool): If ``True``, resumes training from
            a saved checkpoint..
        vectorizer (SurnamesVectorizer): Class responsible for converting
            tokens to numbers.
        mask_index (int): Index of the mask token in the vocabulary.
        client (Client): An interface for selecting different models,
            loss functions and optimizers.

    Example:

        >>> # Create the SurnamesDataset from the dataset's csv file
        >>> from surnames_generator.data.datasets import SurnamesDataset
        >>> dataset = SurnamesDataset.load_dataset_from_csv(
        ...     surnames_csv="./path/to/csv/train.csv"
        ... )
        >>>
        >>> # Define training hyperparameters in the following format:
        >>> hyperparameters = {
        ...     "general_hyperparameters": {
        ...         "num_epochs": 100,
        ...         "batch_size": 64,
        ...         "optimizer_name": "adam",
        ...         "model_name": "lstm_generator",
        ...         "loss_name": "nll",
        ...         "lr_patience": 3,
        ...         "lr_reduce_factor": 0.25,
        ...         "ea_patience": 7,
        ...         "ea_delta": 0.005
        ...     },
        ...     "model_init_params": {
        ...         "embedding_dim": 50,
        ...         "hidden_size": 50,
        ...         "dropout": 0.1,
        ...         "num_layers": 1,
        ...         "with_condition": True
        ...     },
        ...     "optimizer_init_params": {
        ...         "lr": 0.001
        ...     }
        ... }
        >>>
        >>> # Initialize the training controller
        >>> training_controller = TrainingController(
        ...     dataset=dataset,
        ...     hyperparameters=hyperparameters,
        ...     checkpoints_dir="./checkpoints/"
        ... )
        >>>
        >>> # Start training process
        >>> training_controller.prepare_and_start_training()
    """

    def __init__(
        self,
        dataset: SurnamesDataset,
        hyperparameters: Dict[str, Any],
        checkpoints_dir: str,
        resume: bool = False,
    ) -> None:
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.checkpoints_dir = checkpoints_dir
        self.resume = resume

        self.vectorizer = dataset.get_vectorizer()
        self.mask_index = self.vectorizer.char_vocab.mask_index
        self.client = Client(init_params=hyperparameters, vectorizer=self.vectorizer)

    def _load_training_params(self) -> Dict[str, Any]:
        """Returns all the training hyperparameters from the
        configuration file."""
        training_params = {
            "num_rows": len(self.dataset.surnames_df),
            **self.hyperparameters["general_hyperparameters"],
            **self.hyperparameters["model_init_params"],
            **self.hyperparameters["optimizer_init_params"],
        }
        training_params.update(
            {
                "mode": (
                    "conditioned"
                    if self.hyperparameters["model_init_params"]["with_condition"]
                    else "unconditioned"
                )
            },
        )
        return training_params

    def _create_dataloaders(self, train_ids, val_ids) -> Tuple[DataLoader, DataLoader]:
        """Creates training and validation dataloaders from
        a sample of training and validation indices.

        Args:
            train_ids: Sample of training indices.
            val_ids: Sample of validation indices.

        Returns:
            Tuple[DataLoader, DataLoader]: The training
                and validation dataloaders.
        """
        train_dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.hyperparameters["general_hyperparameters"]["batch_size"],
            drop_last=True,
            sampler=SubsetRandomSampler(train_ids),
        )
        val_dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.hyperparameters["general_hyperparameters"]["batch_size"],
            drop_last=True,
            sampler=SubsetRandomSampler(val_ids),
        )

        return train_dataloader, val_dataloader

    def prepare_and_start_training(self) -> None:
        """Orchestrates the training process."""
        set_seeds(seed=42)
        training_params = self._load_training_params()

        model = self.client.model_client(
            name=self.hyperparameters["general_hyperparameters"]["model_name"]
        )
        optimizer = self.client.optimizer_client(
            name=self.hyperparameters["general_hyperparameters"]["optimizer_name"],
            model_params=model.parameters(),
        )
        loss_fn = self.client.loss_client(
            name=self.hyperparameters["general_hyperparameters"]["loss_name"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=self.hyperparameters["general_hyperparameters"]["lr_reduce_factor"],
            patience=self.hyperparameters["general_hyperparameters"]["lr_patience"],
        )
        checkpoint_path = os.path.join(
            self.checkpoints_dir,
            f"{model.__class__.__name__.lower()}_" f"{training_params['mode']}.pth",
        )

        if loss_fn.__class__.__name__ == "NLLLoss":
            apply_softmax = True
        else:
            apply_softmax = False

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            vectorizer=self.vectorizer,
            checkpoint_path=checkpoint_path,
            epochs=self.hyperparameters["general_hyperparameters"]["num_epochs"],
            patience=self.hyperparameters["general_hyperparameters"]["ea_patience"],
            delta=self.hyperparameters["general_hyperparameters"]["ea_delta"],
            resume=self.resume,
            mask_index=self.mask_index,
            apply_softmax=apply_softmax,
        )
        with mlflow.start_run():
            mlflow.log_params(params=training_params)

            ssf = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

            logger.info(
                "-------------------------- Training --------------------------"
            )
            logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

            try:
                for fold, (train_ids, val_ids) in enumerate(
                    ssf.split(X=self.dataset, y=self.dataset.surnames_df.nationality)
                ):
                    train_dataloader, val_dataloader = self._create_dataloaders(
                        train_ids=train_ids, val_ids=val_ids
                    )
                    logger.info(
                        f"Training on {len(train_dataloader)} batches of "
                        f"{train_dataloader.batch_size} samples."
                    )
                    logger.info(
                        f"Evaluating on {len(val_dataloader)} batches of "
                        f"{val_dataloader.batch_size} samples."
                    )

                    with Timer() as t:
                        trainer.train(
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                        )
                    logger.info(f"Training took {t.elapsed} seconds.\n")
            except KeyboardInterrupt:
                logger.info("Exiting loop.")
            finally:
                plt.show()


class SurnamesGeneratorController:
    """Controls the flow of generating surnames.

    Supports generation of random surnames (unconditioned
    generation) or surnames biased towards a provided
    nationality (conditioned generation).

    The :attr:`model_init_params` dictionary must contain
    a boolean key, i.e., `with_condition`, that determines
    whether to generate random surnames, or surnames, biased
    towards a nationality.

    Args:
        model_path (str): Path to the trained model.
        model_name (str): The name of the model to initialize
            and load its save `state_dict`.
        model_init_params (Dict[str, Any]): The parameters to
            initialize the model with.
        vectorizer (CharacterVectorizer): The dataset's vectorizer.
        num_samples (int, optional). Number of surnames to generate
            (default=5).

    Attributes:
        model_path (str): Path to the trained model.
        model_name (str): The name of the model to initialize
            and load its save `state_dict`.
        model_init_params (Dict[str, Any]): The parameters to
            initialize the model with.
        vectorizer (CharacterVectorizer): The dataset's vectorizer.
        num_samples (int). Number of surnames to generate.
        with_condition (bool). If ``True``, generates surnames based
            on dataset's nationalities. Otherwise, generates random
            surnames.
        surnames_generator (SurnamesGenerator): Generates surnames.
        client (Client): An interface for selecting different models.
        model (torch.nn.Module): The trained model.

    .. note::
        If :attr:`with_condition` is ``True`` the generator will
        produce a ``num_samples`` of surnames for each nationality
        in the dataset.

    .. note::
        The number of surnames to generate is defined in the evaluation
        configuration (`yaml`) file.

    Example:

        >>> # Define models initialization parameters.
        >>> # In our case they are the same for all the models.
        >>> model_init_params = {
        ...     "embedding_dim": 50,
        ...     "hidden_size": 50,
        ...     "dropout": 0.1,
        ...     "num_layers": 1,
        ...     "with_condition": True
        ... }
        >>>
        >>> # Initialize the dataset's vectorizer with custom data
        >>> # In practice you need to initialize the vectorizer from
        >>> # the actual data used for the training of the model.
        >>> df = pd.DataFrame(data={
        ...     "nationality": ["Russian", "English", "Irish"],
        ...     "surname": ["Shakhmagon", "Verney", "Ruadhain"]
        ... })
        >>> df
          nationality     surname
        0     Russian  Shakhmagon
        1     English      Verney
        2       Irish    Ruadhain
        >>>
        >>> vectorizer = CharacterVectorizer.from_dataframe(surnames_df=df)
        >>>
        >>> # Initialize the generator controller
        >>> generator_controller = SurnamesGeneratorController(
        ...     model_path="./checkpoints/lstmgenerator_conditioned.pth",
        ...     model_name="lstm_generator",
        ...     model_init_params=model_init_params,
        ...     vectorizer=vectorizer,
        ...     num_samples=10
        ... )
        >>>
        >>> # Generate conditioned surnames (since with_condition=True
        >>> # in model's initialization parameters)
        >>> generator_controller.generate_surnames()
        2024-03-19 18:40:21,882:  INFO      Samples for English:
        2024-03-19 18:40:21,884:  INFO      	- Alloy
        2024-03-19 18:40:21,884:  INFO      	- Ellam
        2024-03-19 18:40:21,884:  INFO      	- Shoroe
        2024-03-19 18:40:21,884:  INFO      Samples for Russian:
        2024-03-19 18:40:21,885:  INFO      	- Adenov
        2024-03-19 18:40:21,885:  INFO      	- Nijneson
        2024-03-19 18:40:21,885:  INFO      	- Pivirigakov
        2024-03-19 18:40:21,885:  INFO      Samples for Japanese:
        2024-03-19 18:40:21,886:  INFO      	- Sasjo
        2024-03-19 18:40:21,887:  INFO      	- Hiuishe
        2024-03-19 18:40:21,887:  INFO      	- Namaha
        2024-03-19 18:40:21,887:  INFO      Samples for Spanish:
        2024-03-19 18:40:21,888:  INFO      	- Canntiar
        2024-03-19 18:40:21,888:  INFO      	- Berteri
        2024-03-19 18:40:21,888:  INFO      	- Allo
    """

    def __init__(
        self,
        model_path: str,
        model_name: str,
        model_init_params: Dict[str, Any],
        vectorizer: CharacterVectorizer,
        num_samples: int = 5,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.model_init_params = model_init_params
        self.vectorizer = vectorizer
        self.num_samples = num_samples

        self.with_condition: bool = model_init_params["with_condition"]
        self.surnames_generator: SurnamesGenerator = SurnamesGenerator(
            vectorizer=vectorizer
        )
        self.client: Client = Client(
            init_params={"model_init_params": model_init_params},
            vectorizer=self.vectorizer,
        )

        self.model: torch.nn.Module = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """Initializes a model based on the :attr:`model_name` and
        loads its save `state_dict` from a filepath."""
        model = self.client.model_client(name=self.model_name)
        checkpoint = torch.load(f=self.model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def _generate_random_surnames(self) -> None:
        """Implements unconditioned surname generation."""
        surnames = self.surnames_generator.generate_unconditioned_surnames(
            model=self.model, num_samples=self.num_samples
        )
        for surname in surnames:
            logger.info(surname)

    def _generated_surnames_per_nationality(self) -> None:
        """Implements conditioned surname generation."""
        for index in range(len(self.vectorizer.category_vocab)):
            nationality = self.vectorizer.category_vocab.lookup_index(index)
            logger.info(f"Samples for {nationality}:")
            surnames = self.surnames_generator.generate_conditioned_surnames(
                model=self.model, nationality_idx=index, num_samples=self.num_samples
            )
            for surname in surnames:
                logger.info(f"\t- {surname}")

    def generate_surnames(self) -> None:
        """Decides whether to implement conditioned or unconditioned
        surname generation based on the value of :attr:`with_condition`.
        """
        if self.with_condition:
            self._generated_surnames_per_nationality()
        else:
            self._generate_random_surnames()
