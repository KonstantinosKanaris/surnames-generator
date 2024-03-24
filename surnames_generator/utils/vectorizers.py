from __future__ import annotations

from typing import Tuple

import pandas as pd
import torch

from surnames_generator.utils.vocabulary import SequenceVocabulary, Vocabulary


class CharacterVectorizer:
    """Responsible for implementing character-level based
    vectorization of an input text sequence.

    Each character in the input text sequence is substituted
    with its corresponding index in the character vocabulary.

    The input text sequence is converted into two 1-D tensors
    of integers. Refer to the ``vectorize()`` method.

    Args:
        char_vocab (SequenceVocabulary): Vocabulary constructed
            from dataset's collection of text characters.
        category_vocab (Vocabulary): Vocabulary constructed
            from dataset's categories.

    Example:

        >>> # Create data (surnames dataset)
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
        >>> # Create the vectorizer
        >>> vectorizer = CharacterVectorizer.from_dataframe(surnames_df=df)

        >>> # Vectorize a surname
        >>> observations, targets = vectorizer.vectorize(text="Verney")
        >>> observations
        tensor([ 2, 12, 13, 14, 11, 13, 15,  0])
        >>> targets
        tensor([12, 13, 14, 11, 13, 15,  3,  0])
    """

    def __init__(
        self, char_vocab: SequenceVocabulary, category_vocab: Vocabulary
    ) -> None:
        self.char_vocab: SequenceVocabulary = char_vocab
        self.category_vocab: Vocabulary = category_vocab

    def vectorize(
        self, text: str, vector_length=-1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Converts an input text sequence into two 1-D tensors
        of integers.

        The first tensor is for the token observations (`X`), and
        consist of all character indexes of the input text sequence
        except the last one. The second tensor is for the token targets
        (`y`), and consist of all characters of the input text sequence
        except the first one.

        Args:
            text (str): The text sequence to be vectorized into a tensor
                of observations and a tensor of targets.
            vector_length (int, optional): Size of the output 1-D tensors . -1 means
                the length of the input tokenized text (default=-1).

        Returns: observations, targets
            * **observations**: tensor of shape :math:`(L)`
            * **targets**: tensor of shape :math:`(L)`

            where:

            .. math::
                \begin{aligned}
                    L ={} & \text{sequence length} \\
                \end{aligned}
        """
        indices = [self.char_vocab.begin_seq_index]
        indices.extend(self.char_vocab.lookup_token(token) for token in text)
        indices.append(self.char_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        observations = torch.empty(size=(vector_length,), dtype=torch.int64)
        from_indices = indices[:-1]
        observations[: len(from_indices)] = torch.tensor(
            from_indices, dtype=torch.int64
        )  # noqa: E203
        observations[len(from_indices) :] = self.char_vocab.mask_index  # noqa: E203

        targets = torch.empty(size=(vector_length,), dtype=torch.int64)
        to_indices = indices[1:]
        targets[: len(to_indices)] = torch.tensor(
            to_indices, dtype=torch.int64
        )  # noqa: E203
        targets[len(to_indices) :] = self.char_vocab.mask_index  # noqa: E203

        return observations, targets

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
    ) -> CharacterVectorizer:
        """Instantiates the `CharacterVectorizer` from a dataframe.

        Args:
            df (pd.DataFrame): The dataset's dataframe.

        Returns:
            Vectorizer: An instance of the `CharacterVectorizer`.
        """
        char_vocab = SequenceVocabulary()
        category_vocab = Vocabulary()

        for category, text in df.values:
            for char in text:
                char_vocab.add_token(token=char)
            category_vocab.add_token(category)

        return cls(char_vocab, category_vocab)
