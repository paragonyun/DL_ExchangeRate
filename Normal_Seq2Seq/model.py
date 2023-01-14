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
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
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
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=input_size)

    def forward(self, x, encoder_input_hidden_state):
        lstm_output, self.hidden = self.lstm(x.unsqueeze(-1), encoder_input_hidden_state)
        output = self.linear(lstm_output)

        return output, self.hidden


class NormalSeq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NormalSeq2SeqModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = Decoder(input_size=input_size, hidden_size=hidden_size)

    def forward(self, 
                inputs, # X
                target_len): # OW
        bs = inputs.shape[0]
        input_size = inputs.shape[2]

        outputs = torch.zeros(bs, target_len, input_size)

        _, hidden_state = self.encoder(inputs)

        decoder_input = inputs[:, -1, :] # 최초 Decoder Input

        ## Decoder (예상값 출력)
        for t in range(target_len): # OW=7이므로 7개의 out을 뱉습니다.
            output, hidden_state = self.decoder(decoder_input, hidden_state)

            output = output.squeeze(1)

            # t시점의 output은 t+1 시점의 Input중 하나로 들어갑니다.
            decoder_input = output

            # 결과 저장
            outputs[:, t, :] = output

        return outputs

    def predict(self, inputs, target_len) :
        self.eval() # Inference Mode

        inputs = inputs.unsqeeze(0)
        bs = inputs.shape[0]
        input_size = inputs.shape[2] # 7이 될 겁니다.
        
        outputs = torch.zeros(bs, target_len, input_size)

        _, hidden_state = self.encoder(inputs)
        decoder_input = inputs[:, -1, :]

        for t in range(target_len):
            output, hidden_state = self.decoder(decoder_input, hidden_state)

            output = output.squeeze(1)

            decoder_input = output

            outputs[:, t, :] = output
        
        return outputs.detach().numpy()[0, : , 0]