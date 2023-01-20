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

        self.encoder_weight = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.decoder_weight = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.value_weight = nn.Linear(in_features=self.hidden_size, out_features=14, bias=False)

        self.fin_linear = nn.Linear(in_features=self.hidden_size + 14, out_features=1)

    def forward(self, x, encoder_input_hidden_state):
        
        lstm_output, self.hidden = self.lstm(
            x.unsqueeze(-1), encoder_input_hidden_state
        )

        ## hidden state활용
        attn_scores = self.value_weight(
            torch.tanh(
                self.encoder_weight(encoder_input_hidden_state[0].permute(1, 0, 2)) + 
                self.decoder_weight(self.hidden[0].permute(1, 0, 2))
            )
        )
        attn_weight = torch.softmax(attn_scores, dim=2)

        context_vector = torch.cat((attn_weight, lstm_output), dim=2).squeeze(1)

        output = self.fin_linear(context_vector)
        
        return output, self.hidden, attn_weight


class AttentionSeq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionSeq2SeqModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = Decoder(input_size=input_size, hidden_size=hidden_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, inputs, target_len):  # X  # OW
        bs = inputs.shape[0]
        input_size = inputs.shape[2]

        outputs = torch.zeros(bs, target_len, input_size)

        _, hidden_state = self.encoder(inputs)

        decoder_input = inputs[:, -1, :]  # 최초 Decoder Input

        
        total_atten_weight = torch.zeros(bs, 1, 14).to(self.device)
        ## Decoder (예상값 출력)
        for t in range(target_len):  # OW=7이므로 7개의 out을 뱉습니다.
            output, hidden_state, attn_weight = self.decoder(decoder_input, hidden_state)

            #output = output.squeeze(1)

            # t시점의 output은 t+1 시점의 Input중 하나로 들어갑니다.
            decoder_input = output

            # 결과 저장
            outputs[:, t, :] = output

            # 종합적인 attn weight를 확인
            total_atten_weight += attn_weight

        return outputs, total_atten_weight

    def predict(self, inputs, target_len):
        self.eval()  # Inference Mode

        inputs = inputs.unsqueeze(0)
        bs = inputs.shape[0]
        input_size = inputs.shape[2]  # 7이 될 겁니다.

        outputs = torch.zeros(bs, target_len, input_size)

        _, hidden_state = self.encoder(inputs)
        decoder_input = inputs[:, -1, :]

        total_attn_weight = torch.zeros(bs, 1, 14)

        for t in range(target_len):
            output, hidden_state, attn_weight = self.decoder(decoder_input, hidden_state)

            # output = output.squeeze(1)

            decoder_input = output

            outputs[:, t, :] = output
            
            total_attn_weight += attn_weight

        return outputs.detach().numpy()[0, :, 0], total_attn_weight
