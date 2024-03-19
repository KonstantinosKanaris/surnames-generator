from __future__ import annotations

from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from surnames_generator.utils.vectorizers import CharacterVectorizer


class SurnamesDataset(Dataset):
    """Surnames Dataset.

    Responsible for providing access to the dataset's data
    points. Each data point is returned in vectorized form
    and consists of token observations `X`, token targets `y`,
    and the surname's nationality index.

    Refer to `surnames_generator.utils.vectorizer.CharacterVectorizer`
    for  information on how the vectorization process takes place.

    Args:
        surnames_df (pd.DataFrame): Dataset's dataframe.
        vectorizer (CharacterVectorizer): Dataset's
            vectorizer.

    Example:

        >>> # Create similar data for the example
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
        >>> # Create vectorizer
        >>> from surnames_generator.utils.vectorizers import CharacterVectorizer
        >>> vectorizer = CharacterVectorizer.from_dataframe(surnames_df=df)
        >>>
        >>> # Create the Dataset
        >>> dataset = SurnamesDataset(surnames_df=df, vectorizer=vectorizer)
        >>>
        >>> # Random data point
        >>> random_idx = torch.randint(low=0, high=len(dataset), size=[1,]).item()
        >>> observations, targets, nationality_idx = dataset[random_idx]
        >>> observations
        tensor([ 2,  4,  5,  6,  7,  5,  8,  6,  9, 10, 11,  0])
        >>> targets
        tensor([ 4,  5,  6,  7,  5,  8,  6,  9, 10, 11,  3,  0])
        >>> nationality_idx
        tensor(0)
    """

    def __init__(
        self, surnames_df: pd.DataFrame, vectorizer: CharacterVectorizer
    ) -> None:
        self.surnames_df: pd.DataFrame = surnames_df
        self._vectorizer: CharacterVectorizer = vectorizer

        # + 2 for the <BEGIN> and <END> tokens
        self._max_seq_len = surnames_df.surname.str.len().max() + 2

    def get_vectorizer(self) -> CharacterVectorizer:
        """Returns the vectorizer."""
        return self._vectorizer

    def get_num_batches(self, batch_size) -> int:
        """Given a batch size, returns the number of batches
        in the dataset.

        Args:
            batch_size (int): The batch size.

        Returns:
            int: The number of batches in the dataset.
        """
        return len(self) // batch_size

    @classmethod
    def load_dataset_from_csv(cls, surnames_csv: str) -> SurnamesDataset:
        """Initializes the `SurnamesDataset` from a csv file.

        Args:
            surnames_csv (str): Location of dataset's csv file.

        Returns:
            SurnamesDataset: An instance of `SurnamesDataset`.
        """
        surnames_df = pd.read_csv(filepath_or_buffer=surnames_csv)
        return cls(
            surnames_df=surnames_df,
            vectorizer=CharacterVectorizer.from_dataframe(df=surnames_df),
        )

    def __len__(self) -> int:
        """Returns the length of the surnames' dataset."""
        return len(self.surnames_df)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Get dataset's data point in vectorized form based
        on the input index.

        The data point consists of the token observations (`X`),
        the token targets (`y`) and the nationality index.

        Refer to `surnames_generator.utils.vectorizer.SurnamesVectorizer`.

        Args:
            index (int): The index to the data point.

        Returns: observations, targets, nationality_index

            * **observations**: tensor of shape :math:`(L)`
            * **targets**: tensor of shape :math:`(L)`
            * **nationality_index**: tensor of shape :math: `()`

            where:

            .. math::
                \begin{aligned}
                    L ={} & \text{sequence length} \\
                \end{aligned}
        """
        row = self.surnames_df.iloc[index]

        observations, targets = self._vectorizer.vectorize(
            text=row.surname, vector_length=self._max_seq_len
        )

        nationality_index = self._vectorizer.category_vocab.lookup_token(
            token=row.nationality
        )

        return (
            observations,
            targets,
            torch.tensor(nationality_index, dtype=torch.int64),
        )
