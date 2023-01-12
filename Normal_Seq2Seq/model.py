import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Cell은 LSTM으로 했습니다."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
    ):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(self, x):
        lstm_output, self.hidden = self.lstm(x)

        return lstm_output, self.hidden


class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
    ):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=input_size)

    def forward(self, x):
        lstm_output, self.hidden = self.lstm(x)
        output = self.linear(lstm_output)

        return output, self.hidden


class NormalSeq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NormalSeq2SeqModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = Encoder(input_size=self.input_size, hidden_size=self.hidden_size)
        self.decoder = Decoder(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, x):

        return x
