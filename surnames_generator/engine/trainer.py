from typing import Tuple

import mlflow
import torch
import torch.utils.data
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

from surnames_generator import logger
from surnames_generator.engine.generator import SurnamesGenerator
from surnames_generator.engine.utils import EarlyStopping, normalize_tensors
from surnames_generator.utils.aux import load_general_checkpoint
from surnames_generator.utils.vectorizers import CharacterVectorizer


class Trainer:
    """Class for training a character-based seq-to-seq text
    generation PyTorch model.

    Incorporates functionalities such as early stopping,
    learning rate reduction, resume from checkpoint, and
    MLFlow tracking.

    Args:
        checkpoint_path (str): The file path to save or load a
            checkpoint.
        model (torch.nn.Module): The PyTorch model to be trained.
        loss_fn (torch.nn.Module): The loss function used for
            optimization.
        optimizer (torch.optim.Optimizer): The optimizer used for
            updating model parameters.
        scheduler: Reduces the learning rate when the validation
            loss stops improving.
        vectorizer (CharacterVectorizer): Vectorizes a text sequence
            to observations `X` and targets `y`.
        epochs (int, optional): Number of training epochs
            (default=5).
        patience (int, optional): Number of epochs to wait before
        early stopping (default=5).
        delta (float, optional): Minimum change in monitored quantity
            to qualify as an improvement (default=0).
        resume (bool, optional): If ``True``, resumes training from
            the specified checkpoint. Defaults to ``False``.
        mask_index (int, optional): The masking index to ignore during
            the calculation of loss and accuracy metrics (default=0).
        apply_softmax (bool, optional): If ``True`` the model applies
            `nn.LogSoftmax()` to the prediction logits during the
            forward pass. Defaults to ``False``.
    """

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        vectorizer: CharacterVectorizer,
        checkpoint_path: str,
        epochs: int = 5,
        patience: int = 5,
        delta: float = 0,
        resume: bool = False,
        mask_index: int = 0,
        apply_softmax: bool = False,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.vectorizer = vectorizer
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.resume = resume
        self.mask_index = mask_index
        self.apply_softmax = apply_softmax

        self.early_stopping: EarlyStopping = EarlyStopping(
            patience=patience, delta=delta, path=checkpoint_path, verbose=True
        )
        self.surnames_generator: SurnamesGenerator = SurnamesGenerator(
            vectorizer=vectorizer
        )

    def compute_accuracy(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> float:
        """Returns the computed accuracy value.

        Args:
            y_pred (torch.Tensor): The output of the model.
            y_true (torch.Tensor): The target predictions.
        """
        y_pred, y_true = normalize_tensors(y_pred=y_pred, y_true=y_true)
        y_pred_indices = torch.argmax(y_pred, dim=1)

        correct_indices = torch.eq(input=y_pred_indices, other=y_true).float()
        valid_indices = torch.ne(input=y_true, other=self.mask_index).float()

        n_correct = (correct_indices * valid_indices).sum().item()
        n_valid = valid_indices.sum().item()

        return n_correct / n_valid * 100

    def sequence_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Normalizes the tensor sizes and calculates the
        loss of the sequence predictions.

        Args:
            y_pred (torch.Tensor): The output of the model.
            y_true (torch.Tensor): The target predictions.

        Return:
            torch.Tensor: The calculated sequence loss.
        """
        y_pred, y_true = normalize_tensors(y_pred=y_pred, y_true=y_true)
        return self.loss_fn(y_pred, y_true)

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """Trains a seq-to-seq text generation PyTorch model.

        The training routine expects from the dataloaders to provide
        two sequence of integers which represent the token observations,
        and the token targets at each time step.

        Performs the training using the provided dataloaders, loss function,
        and optimizer. It also performs evaluation on the validation data at
        the end of each epoch. Checkpointing is supported, optionally allowing
        for the resumption of training from a saved checkpoint.

        The training process includes learning rate reduction and early stopping
        to prevent over-fitting. The training loop stops if the validation loss
        does not improve for a certain number of epochs, defined from the
        :attr:`patience` class attribute.

        Args:
            train_dataloader (torch.utils.data.DataLoader):
                A `DataLoader` instance for providing batches of
                training data.
            val_dataloader (torch.utils.data.DataLoader):
                A `DataLoader` instance for providing batches of
                validation data.
        """
        self.model.to(self.__class__.DEVICE)

        epoch_bar = tqdm(desc="Training routine", total=self.epochs, position=0)
        train_bar = tqdm(
            desc="split=train", total=len(train_dataloader), position=1, leave=True
        )
        val_bar = tqdm(
            desc="split=val", total=len(val_dataloader), position=1, leave=True
        )

        start_epoch = 0
        if self.resume:
            checkpoint = load_general_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                filepath=self.checkpoint_path,
            )
            self.model = checkpoint["model"].to(self.__class__.DEVICE)
            self.optimizer = checkpoint["optimizer"]
            loss_value = checkpoint["val_loss"]
            start_epoch = checkpoint["epoch"] + 1

            logger.info(
                f"Resume training from general checkpoint: {self.checkpoint_path}."
            )
            logger.info(f"Last training loss value: {loss_value:.4f}")
            logger.info(f"Resuming from {start_epoch + 1} epoch...")

        for epoch_index in range(start_epoch, self.epochs):
            train_loss, train_acc = self._train_step(
                dataloader=train_dataloader,
                tqdm_bar=train_bar,
                epoch_index=epoch_index,
            )
            val_loss, val_acc = self._val_step(
                dataloader=val_dataloader,
                tqdm_bar=val_bar,
                epoch_index=epoch_index,
            )

            print("\n")
            logger.info(
                f"===>>> epoch: {epoch_index + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.4f}"
            )

            mlflow.log_metrics(
                metrics={
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch_index,
            )

            self.early_stopping(
                epoch=epoch_index,
                model=self.model,
                optimizer=self.optimizer,
                val_loss=val_loss,
            )

            self.scheduler.step(val_loss)

            # move model to cpu for generating sample of surnames
            self.model.cpu()
            sampled_surnames = self.surnames_generator.generate_unconditioned_surnames(
                model=self.model, num_samples=2
            )
            epoch_bar.set_postfix(
                epoch=epoch_index,
                lr=self.scheduler.get_last_lr(),
                sample1=sampled_surnames[0],
                sample2=sampled_surnames[1],
            )
            # move model back to target device to resume training
            self.model.to(self.__class__.DEVICE)

            if self.early_stopping.early_stop:
                logger.info("Training stopped due to early stopping.")
                break
            else:
                train_bar.n, val_bar.n = 0, 0
                epoch_bar.update()
                continue

    def _train_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        tqdm_bar: tqdm,
        epoch_index: int,
    ) -> Tuple[float, float]:
        """Trains a seq-to-seq text generation PyTorch model
        for a single epoch.

        Turns the target model to `train` mode and then runs
        through all the required training steps (forward pass,
        loss calculation, optimizer step).

        Args:
            dataloader (torch.utils.data.DataLoader):
                A `DataLoader` instance for providing batches of
                training data.
            tqdm_bar (tqdm): Custom tqdm bar for the training
                process.
            epoch_index (int): Current epoch.

        Returns:
          Tuple[float, float]: The training loss and accuracy.
        """
        self.model.train()
        train_loss, train_acc = 0.0, 0.0
        for batch_idx, (X, y, category_idx) in enumerate(
            BackgroundGenerator(dataloader)
        ):
            X, y = X.to(self.__class__.DEVICE), y.to(self.__class__.DEVICE)
            category_idx = category_idx.to(self.__class__.DEVICE)

            y_pred = self.model(X, category_idx, apply_softmax=self.apply_softmax)
            loss = self.sequence_loss(y_pred=y_pred, y_true=y)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            # plot_grad_flow(self.model.named_parameters())
            self.optimizer.step()

            acc = self.compute_accuracy(y_pred=y_pred, y_true=y)
            train_acc += acc

            tqdm_bar.set_postfix(epoch=epoch_index + 1, batch=batch_idx + 1)
            tqdm_bar.update()

        train_loss /= len(dataloader)
        train_acc /= len(dataloader)

        return train_loss, train_acc

    def _val_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        tqdm_bar: tqdm,
        epoch_index: int,
    ) -> Tuple[float, float]:
        """Validates a seq-to-seq text generation PyTorch model
        for a single epoch.

        Turns the target model to `eval` model and then performs
        a forward pass on the validation data.

        Args:
            dataloader (torch.utils.data.DataLoader):
                A `DataLoader` instance for providing batches of
                validation data.
            tqdm_bar (tqdm): Custom tqdm bar for the validation
                process.
            epoch_index (int): Current epoch.

        Returns:
          Tuple[float, float]: The validation loss and accuracy.
        """
        self.model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.inference_mode():
            for batch_idx, (X, y, category_idx) in enumerate(
                BackgroundGenerator(dataloader)
            ):
                X, y = X.to(self.__class__.DEVICE), y.to(self.__class__.DEVICE)
                category_idx = category_idx.to(self.__class__.DEVICE)

                y_pred = self.model(X, category_idx, apply_softmax=self.apply_softmax)
                loss = self.sequence_loss(y_pred=y_pred, y_true=y)
                val_loss += loss.item()

                acc = self.compute_accuracy(y_pred=y_pred, y_true=y)
                val_acc += acc

                tqdm_bar.set_postfix(epoch=epoch_index + 1, batch=batch_idx + 1)
                tqdm_bar.update()

        val_loss /= len(dataloader)
        val_acc /= len(dataloader)

        return val_loss, val_acc
