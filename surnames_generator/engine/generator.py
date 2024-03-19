from typing import List

import torch

from surnames_generator.utils.vectorizers import CharacterVectorizer


class SurnamesGenerator:
    """Generates random surnames or surnames biased
    towards a provided nationality.

    Args:
        vectorizer (CharacterVectorizer): The surnames vectorizer for
            accessing the vocabulary.

    Attributes:
        vectorizer (CharacterVectorizer): The surnames vectorizer for
            accessing the vocabulary.
    """

    def __init__(self, vectorizer: CharacterVectorizer) -> None:
        self.vectorizer = vectorizer

    def unconditioned_sample_generation(
        self,
        model: torch.nn.Module,
        num_samples: int = 1,
        sample_size: int = 20,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        r"""Generates a number of sequences consisting of character
        indices.

        Each sequence of indices represents a random surname.

        Args:
            model (torch.nn.Module): The trained model to generate indices.
            num_samples (int, optional): Batch size. Number of sequences to
                generate (default=1).
            sample_size (int, optional): The length of the generated sequences
                (default=20).
            temperature (float, optional): Accentuates of flattens the distribution.
                0.0 < temperature < 1.0 makes the distribution peakier.
                temperature > 1.0 makes the distribution more uniform.

        Outputs: indices
            * **indices**: tensor of shape :math:`(N, L)`.

            where:

            .. math::
                \begin{aligned}
                    N ={} & \text{batch size} \\
                    L ={} & \text{sequence length} \\
                \end{aligned}
        """
        begin_seq_index = [
            self.vectorizer.char_vocab.begin_seq_index for _ in range(num_samples)
        ]
        begin_seq_index_tensor = torch.tensor(
            data=begin_seq_index, dtype=torch.int64
        ).unsqueeze(dim=1)

        indices = [begin_seq_index_tensor]
        h_t = None

        model.eval()
        with torch.inference_mode():
            for time_step in range(sample_size):
                x_t = indices[time_step]
                x_emb_t = model.char_emb(x_t)
                rnn_out_t, h_t = model.rnn(x_emb_t, h_t)
                prediction_vector = model.fc(rnn_out_t.squeeze(dim=1))
                probability_vector = torch.softmax(
                    prediction_vector / temperature, dim=1
                )
                indices.append(torch.multinomial(probability_vector, num_samples=1))
        stacked_indices = torch.stack(indices).squeeze().permute(1, 0)
        return stacked_indices

    def conditioned_sample_generation(
        self,
        model: torch.nn.Module,
        nationalities: List[int],
        sample_size: int = 20,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        r"""Generates a number of sequences consisting of character indices.

        Each sequence of indices represents a surname, biased towards the
        given nationality index.

        Args:
            model (torch.nn.Module): The trained model to generate indices.
            nationalities (List[int]): List of indices representing nationalities.
            sample_size (int, optional): The length of the generated sequences
                (default=20).
            temperature (float, optional): Accentuates of flattens the distribution.
                0.0 < temperature < 1.0 makes the distribution peakier.
                temperature > 1.0 makes the distribution more uniform.

        Outputs: indices
            * **indices**: tensor of shape :math:`(N, L)`.

            where:

            .. math::
                \begin{aligned}
                    N ={} & \text{batch size} \\
                    L ={} & \text{sequence length} \\
                \end{aligned}
        """
        num_samples = len(nationalities)
        begin_seq_index = [
            self.vectorizer.char_vocab.begin_seq_index for _ in range(num_samples)
        ]
        begin_seq_index_tensor = torch.tensor(
            data=begin_seq_index, dtype=torch.int64
        ).unsqueeze(dim=1)

        indices = [begin_seq_index_tensor]
        nationality_indices = torch.tensor(
            data=nationalities, dtype=torch.int64
        ).unsqueeze(dim=0)

        cat_emb_t = model.cat_emb(nationality_indices)
        if model.__class__.__name__.lower().startswith("lstm"):
            cat_hidden_t, cat_cell_t = torch.split(cat_emb_t, model.hidden_size, dim=-1)

        model.eval()
        with torch.inference_mode():
            for time_step in range(sample_size):
                x_t = indices[time_step]
                x_emb_t = model.char_emb(x_t)
                if model.__class__.__name__.lower().startswith("lstm"):
                    rnn_out_t, (cat_hidden_t, cat_cell_t) = model.rnn(
                        x_emb_t, (cat_hidden_t, cat_cell_t)
                    )
                else:
                    rnn_out_t, cat_emb_t = model.rnn(x_emb_t, cat_emb_t)
                prediction_vector = model.fc(rnn_out_t.squeeze(dim=1))
                probability_vector = torch.softmax(
                    prediction_vector / temperature, dim=1
                )
                indices.append(torch.multinomial(probability_vector, num_samples=1))

        stacked_indices = torch.stack(indices).squeeze().permute(1, 0)
        return stacked_indices

    def decode_samples(self, sampled_indices: torch.Tensor) -> List[str]:
        """Transforms the generated sequences of indices to
        readable surnames.

        Args:
            sampled_indices (torch.Tensor): The generated sample of
                sequence indices.

        Returns:
            List[str]: The generated surnames.
        """
        decoded_surnames = []
        vocab = self.vectorizer.char_vocab

        for sample_index in range(sampled_indices.size(0)):
            surname = ""
            for time_step in range(sampled_indices.size(1)):
                sample_item = sampled_indices[sample_index, time_step].item()
                if sample_item == vocab.begin_seq_index:
                    continue
                elif sample_item == vocab.end_seq_index:
                    break
                else:
                    surname += vocab.lookup_index(int(sample_item))
            decoded_surnames.append(surname)
        return decoded_surnames

    def generate_unconditioned_surnames(
        self,
        model: torch.nn.Module,
        num_samples: int = 1,
        sample_size: int = 20,
        temperature: float = 1.0,
    ) -> List[str]:
        """Generates a sample of random surnames.

        Args:
            model (torch.nn.Module): The trained model to generate indices.
            num_samples (int, optional): Number of surnames to generate
                (default=1).
            sample_size (int, optional): The length of the generated surnames
                (default=20).
            temperature (float, optional): Accentuates of flattens the distribution.
                0.0 < temperature < 1.0 makes the distribution peakier.
                temperature > 1.0 makes the distribution more uniform.

        Returns:
            List[str]: The generated surnames.
        """
        indices = self.unconditioned_sample_generation(
            model=model,
            num_samples=num_samples,
            sample_size=sample_size,
            temperature=temperature,
        )
        surnames = self.decode_samples(sampled_indices=indices)
        return surnames

    def generate_conditioned_surnames(
        self,
        model: torch.nn.Module,
        nationality_idx: int,
        num_samples: int = 1,
        sample_size: int = 20,
        temperature: float = 1.0,
    ) -> List[str]:
        """Generates a sample of surnames, biased towards the
        provided nationality index.

        Args:
            model (torch.nn.Module): The trained model to generate indices.
            nationality_idx (int): The nationality index.
            sample_size (int, optional): The length of the generated surnames
                (default=20).
            num_samples (int, optional): Number of surnames to generate
                (default=1).
            temperature (float, optional): Accentuates of flattens the distribution.
                0.0 < temperature < 1.0 makes the distribution peakier.
                temperature > 1.0 makes the distribution more uniform.

        Returns:
            List[str]: The generated surnames.
        """
        indices = self.conditioned_sample_generation(
            model=model,
            nationalities=[nationality_idx] * num_samples,
            sample_size=sample_size,
            temperature=temperature,
        )
        surnames = self.decode_samples(sampled_indices=indices)
        return surnames
