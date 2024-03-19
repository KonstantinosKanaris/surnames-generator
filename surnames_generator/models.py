from typing import Optional, Tuple

import torch
from torch import nn


class SurnamesRNNCellGenerator(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        char_vocab_size: int,
        num_categories: int,
        hidden_size: int,
        dropout: float = 0.1,
        padding_idx: int = 0,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.num_categories: int = num_categories
        self.hidden_size: int = hidden_size
        self.batch_first: bool = batch_first

        self.char_emb = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.cat_emb = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=hidden_size
        )

        # sum of the dimensions of the concatenated tensors
        # 1 hidden size for the hidden state and 1 for the category embeddings
        self.i2h = nn.Linear(
            in_features=embedding_dim + 2 * hidden_size, out_features=hidden_size
        )
        self.i2o = nn.Linear(
            in_features=embedding_dim + 2 * hidden_size, out_features=char_vocab_size
        )
        self.o2o = nn.Linear(
            in_features=hidden_size + char_vocab_size, out_features=char_vocab_size
        )

        self.dropout = nn.Dropout(p=dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def initialize_hidden_state(self, batch_size: int) -> torch.Tensor:
        # hidden state batched input size: (1*num_rnn_layers, batch_size, hidden_size)
        return torch.zeros(size=(batch_size, self.hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        nationality_index: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        apply_softmax: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_embedded = self.char_emb(x)
        cat_embedded = self.cat_emb(nationality_index)

        batch_size, feat_size = x_embedded.size()
        if hidden_state is None:
            hidden_state = self.initialize_hidden_state(batch_size=batch_size)

        combined_input = torch.cat(
            tensors=[x_embedded, cat_embedded, hidden_state], dim=1
        )

        hidden = self.i2h(combined_input)
        output = self.i2o(combined_input)
        output_combined = torch.cat([hidden, output], dim=1)
        output = self.dropout(self.o2o(output_combined))

        if apply_softmax:
            output = self.log_softmax(output)

        return output, hidden


class SimpleRNNGenerator(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        char_vocab_size: int,
        num_categories: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        padding_idx: int = 0,
        batch_first: bool = True,
        non_linearity: str = "tanh",
        with_condition: bool = False,
    ) -> None:
        super().__init__()

        if non_linearity not in ["tanh", "relu"]:
            raise ValueError(f"Invalid selected non-linearity: {non_linearity}.")

        self.num_categories: int = num_categories
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.batch_first: bool = batch_first
        self.with_condition: bool = with_condition

        self.char_emb = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.cat_emb = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=hidden_size
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=non_linearity,
            batch_first=batch_first,
        )

        self.fc = nn.Linear(in_features=hidden_size, out_features=char_vocab_size)

        self.dropout = nn.Dropout(p=dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        nationality_index: Optional[torch.Tensor] = None,
        apply_softmax: bool = False,
    ) -> torch.Tensor:
        x_embedded = self.char_emb(x)

        if self.with_condition:
            cat_embedded = self.cat_emb(nationality_index).unsqueeze(dim=0)
            if self.num_layers > 1:
                cat_embedded = cat_embedded.repeat(self.num_layers, 1, 1)
            y_out, _ = self.rnn(x_embedded, cat_embedded)
        else:
            y_out, _ = self.rnn(x_embedded)

        batch_size, seq_size, feat_size = y_out.size()
        # Reshaping 3-D to 2-D because the linear layer requires an input matrix
        y_out = y_out.contiguous().view(batch_size * seq_size, feat_size)

        y_out = self.fc(self.dropout(y_out))

        if apply_softmax:
            y_out = self.log_softmax(y_out)

        new_feat_size = y_out.size(-1)
        y_out = y_out.view(batch_size, seq_size, new_feat_size)

        return y_out


class GRUGenerator(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        char_vocab_size: int,
        num_categories: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        padding_idx: int = 0,
        batch_first: bool = True,
        with_condition: bool = False,
    ) -> None:
        super().__init__()

        self.num_categories: int = num_categories
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.batch_first: bool = batch_first
        self.with_condition: bool = with_condition

        self.char_emb = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.cat_emb = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=hidden_size
        )

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )

        self.fc = nn.Linear(in_features=hidden_size, out_features=char_vocab_size)

        self.dropout = nn.Dropout(p=dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        nationality_index: torch.Tensor,
        apply_softmax: bool = False,
    ) -> torch.Tensor:
        x_embedded = self.char_emb(x)

        if self.with_condition:
            cat_embedded = self.cat_emb(nationality_index).unsqueeze(dim=0)
            y_out, _ = self.rnn(x_embedded, cat_embedded)
        else:
            y_out, _ = self.rnn(x_embedded)

        batch_size, seq_size, feat_size = y_out.size()
        # Reshaping 3-D to 2-D because the linear layer requires an input matrix
        y_out = y_out.contiguous().view(batch_size * seq_size, feat_size)

        y_out = self.fc(self.dropout(y_out))

        if apply_softmax:
            y_out = self.log_softmax(y_out)

        new_feat_size = y_out.size(-1)
        y_out = y_out.view(batch_size, seq_size, new_feat_size)

        return y_out


class LSTMGenerator(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        char_vocab_size: int,
        num_categories: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        padding_idx: int = 0,
        batch_first: bool = True,
        with_condition: bool = False,
    ) -> None:
        super().__init__()

        self.num_categories: int = num_categories
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.batch_first: bool = batch_first
        self.with_condition: bool = with_condition

        self.char_emb = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.cat_emb = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=hidden_size * 2
        )

        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )

        self.fc = nn.Linear(in_features=hidden_size, out_features=char_vocab_size)

        self.dropout = nn.Dropout(p=dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        nationality_index: torch.Tensor,
        apply_softmax: bool = False,
    ) -> torch.Tensor:
        x_embedded = self.char_emb(x)

        if self.with_condition:
            cat_embedded = self.cat_emb(nationality_index).unsqueeze(dim=0)
            cat_hidden, cat_cell = torch.split(cat_embedded, self.hidden_size, dim=-1)
            y_out, _ = self.rnn(x_embedded, (cat_hidden, cat_cell))
        else:
            y_out, _ = self.rnn(x_embedded)

        batch_size, seq_size, feat_size = y_out.size()
        # Reshaping 3-D to 2-D because the linear layer requires an input matrix
        y_out = y_out.contiguous().view(batch_size * seq_size, feat_size)

        y_out = self.fc(self.dropout(y_out))

        if apply_softmax:
            y_out = self.log_softmax(y_out)

        new_feat_size = y_out.size(-1)
        y_out = y_out.view(batch_size, seq_size, new_feat_size)

        return y_out
