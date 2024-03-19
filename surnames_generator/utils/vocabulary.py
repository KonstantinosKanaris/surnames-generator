from __future__ import annotations

from typing import Any, Dict, List, Optional


class Vocabulary:
    """Creates a vocabulary object which maps tokens to indices
    and vice versa.

    Args:
        token_to_idx (Dict[str, int], optional):
            A pre-existing mapping of tokens to indices. Defaults
            to `None`.

    Attributes:
        token_to_idx (Dict[str, int]):
            A dictionary mapping tokens to indices.
        idx_to_token (Dict[int, str]:
            A dictionary mapping indices to tokens.
    """

    def __init__(
        self,
        token_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx: Dict[str, int] = token_to_idx
        self.idx_to_token: Dict[int, str] = {
            idx: token for token, idx in self.token_to_idx.items()
        }

    def to_serializable(self) -> Dict[str, Any]:
        """Returns a dictionary that can be serialized."""
        return {
            "token_to_idx": self.token_to_idx,
        }

    @classmethod
    def from_serializable(cls, contents: Dict[str, Any]) -> Vocabulary:
        """Instantiate a `Vocabulary` object from a serializable
        dictionary.

        Args:
            contents (Dict[str, Any]): The serialized dictionary.

        Returns:
            Vocabulary: A vocabulary object instantiated from
                the provided serialized data.
        """
        return cls(**contents)

    def add_token(self, token: str) -> int:
        """Updates the mapping dictionaries based on the
        provided token.

        Args:
            token (str): The token to be added to the
                vocabulary.

        Returns:
            int: The index corresponding to the token.
        """
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def add_many(self, tokens: List[str]) -> List[int]:
        """Updates the mapping dictionaries based on a
        list of input tokens.

        Args:
            tokens (List[str]): A list of string tokens.

        Returns:
            List[int]: A list of indices corresponding
                to the tokens.
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token: str) -> int:
        """Retrieves the index associated with the token.

        Args:
            token (str): The token to look up.

        Returns:
            int: The index corresponding to the token.
        """
        return self.token_to_idx[token]

    def lookup_index(self, index: int) -> str:
        """Retrieves the token associated with the index.

        Args:
            index (int): The index to look up.

        Returns:
            str: The token corresponding to the index.

        Raises:
            KeyError: If the index is not in the vocabulary.
        """
        if index not in self.idx_to_token:
            raise KeyError(f"The index {index} is not in the vocabulary.")
        return self.idx_to_token[index]

    def __len__(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.token_to_idx)

    def __str__(self) -> str:
        """Returns a string representation of the `Vocabulary` instance."""
        return f"<Vocabulary(size={len(self)})>"


class SequenceVocabulary(Vocabulary):
    """Bundles four special tokens used for sequence data.

    The special tokens are: the `UNK` token, the `MASK` token,
    the `BEGIN-OF-SEQUENCE` token, and the `END-OF-SEQUENCE`
    token where:

    .. code-block:: text

        * `UNK`: The unknown token used for unseen out-of-vocabulary
        input tokens
        * `MASK`: Enables handling variable-length inputs
        * `BEGIN-OF-SEQUENCE`: Start of sequence boundary
        * `END-OF-SEQUENCE`: End of sequence boundary

    Args:
        token_to_idx (Dict[str, int], optional):
            A pre-existing map of tokens to indices. Defaults
            to `None`.
        mask_token (str, optional): The representation of the
            `MASK` token. Defaults to `<MASK>`.
        unk_token (str, optional): The representation of the
            `UNK` token. Defaults to `<UNK>`.
        begin_seq_token (str, optional): The representation of
            the `BEGIN-OF-SEQUENCE` token. Defaults to `<BEGIN>`.
        end_seq_token (str, optional): The representation of
            the `END-OF-SEQUENCE` token. Defaults to `<END>`.

    Attributes:
        token_to_idx (Dict[str, int]):
            A dictionary mapping tokens to indices.
        idx_to_token (Dict[int, str]:
            A dictionary mapping indices to tokens.
        _unk_token (str): The representation of the `UNK` token.
        _mask_token (str): The representation of the `MASK` token.
        _begin_seq_token (str): The representation of the
            `BEGIN-OF-SEQUENCE` token.
        _end_seq_token (str, optional): The representation of the
            `END-OF-SEQUENCE` token.
        unk_index (int): Index associated with the `UNK` token.
        mask_index (int): Index associated with the `MASK` token.
        begin_seq_index (int): Index associated with the
            `BEGIN-OF-SEQUENCE` token.
        end_seq_index (int): Index associated with the
            `END-OF-SEQUENCE` token.
    """

    def __init__(
        self,
        token_to_idx: Optional[Dict[str, int]] = None,
        unk_token: str = "<UNK>",
        mask_token: str = "<MASK>",
        begin_seq_token: str = "<BEGIN>",
        end_seq_token: str = "<END>",
    ) -> None:
        super().__init__(token_to_idx=token_to_idx)
        self._mask_token: str = mask_token
        self._unk_token: str = unk_token
        self._begin_seq_token: str = begin_seq_token
        self._end_seq_token: str = end_seq_token

        self.mask_index = self.add_token(token=mask_token)
        self.unk_index = self.add_token(token=unk_token)
        self.begin_seq_index = self.add_token(token=begin_seq_token)
        self.end_seq_index = self.add_token(token=end_seq_token)

    def to_serializable(self) -> Dict[str, Any]:
        """Returns a dictionary that can be serialized."""
        contents = super().to_serializable()
        contents.update(
            {
                "unk_token": self._unk_token,
                "mask_token": self._mask_token,
                "begin_seq_token": self._begin_seq_token,
                "end_seq_token": self._end_seq_token,
            }
        )
        return contents

    def lookup_token(self, token: str) -> int:
        """Retrieves the index associated with the token or
        the `UNK` index if token isn't found.

        Args:
            token (str): The token to look up.

        Returns:
            int: The index corresponding to the token.

        Notes:
            `unk_index` needs to be >=0 (having been added into
            the `Vocabulary`) for the `UNK` functionality.
        """
        if self.unk_index >= 0:
            return self.token_to_idx.get(token, self.unk_index)
        else:
            return self.token_to_idx[token]
