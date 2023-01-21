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
        self.value_weight = nn.Linear(in_features=self.hidden_size, out_features=1, bias=False)

        self.fin_linear = nn.Linear(in_features=self.hidden_size, out_features=self.input_size)

    def forward(self, x, hidden, encoder_output):
        ## hidden state활용
        # print("input size: ", x.size()) # 32, 1
        # print("encoder output: ", encoder_output.size()) # 32, 14, 64
        # print("hidden[0] size: ", hidden[0].size())
        attn_scores = self.value_weight(
            torch.tanh(
                self.encoder_weight(encoder_output) + 
                self.decoder_weight(hidden[0].permute(1, 0, 2))
            )
        ) 
        # print(attn_scores.size()) # 32, 14, 1
        attn_weight = torch.softmax(attn_scores, dim=1) # 32, 14, 1
        # print(attn_weight.permute(0, 2, 1).size())
        
        context_vector = torch.bmm(attn_weight.permute(0, 2, 1), encoder_output)
        # print("CV : ", context_vector.size()) # 32, 1, 64

        new_input = torch.cat((context_vector, x.unsqueeze(1)), dim=2)

        new_input = new_input.permute(0, 2, 1)
        # print("new input size: ", new_input.size()) # 32, 65, 1

        output, hidden = self.lstm(new_input, hidden) 

        # print("output Size : ", output[:, -1, :].size()) # 32, 64

        fin_output = self.fin_linear(output[:, -1, :])

        return fin_output, hidden, attn_weight


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

        encoder_output, hidden_state = self.encoder(inputs)

        decoder_input = inputs[:, -1, :]  # 최초 Decoder Input

        de_hidden = hidden_state

        
        total_atten_weight = torch.zeros(bs, 14, 1).to(self.device)
        ## Decoder (예상값 출력)
        for t in range(target_len):  # OW=7이므로 7개의 out을 뱉습니다.
            output, de_hidden, attn_weight = self.decoder(decoder_input, hidden=de_hidden, encoder_output=encoder_output)
            # print(output.size())
            #output = output.squeeze(1)

            # t시점의 output은 t+1 시점의 Input중 하나로 들어갑니다.
            decoder_input = output
            # print(output[0])
            # print(output.size())
            
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

        encoder_output, hidden_state = self.encoder(inputs)
        decoder_input = inputs[:, -1, :]

        de_hidden = hidden_state

        total_attn_weight = torch.zeros(bs, 14, 1)

        for t in range(target_len):
            output, hidden_state, attn_weight = self.decoder(decoder_input, de_hidden, encoder_output)

            # output = output.squeeze(1)

            decoder_input = output

            outputs[:, t, :] = output
            
            total_attn_weight += attn_weight

        return outputs.detach().numpy()[0, :, 0], total_attn_weight
