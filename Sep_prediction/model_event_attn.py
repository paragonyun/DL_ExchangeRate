import torch
import torch.nn as nn
import torch.nn.functional as F

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
            dropout=0.3
        )

        self.ln = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        lstm_output, self.hidden = self.lstm(x)
        lstm_output = self.ln(lstm_output)

        return lstm_output, self.hidden


class Decoder(nn.Module):
    def __init__(
        self,
        vectorized_events_mat,
        input_size,
        hidden_size,
        num_layers=1,
    ):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=hidden_size+1+10,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            
        )

        self.encoder_weight = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.decoder_weight = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.value_weight = nn.Linear(in_features=self.hidden_size, out_features=1, bias=False)

        self.fin_linear = nn.Linear(in_features=self.hidden_size, out_features=1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"   

        self.gate = nn.Linear(in_features=10, out_features=10, bias=False)

        self.vectorized_events_mat=vectorized_events_mat

        self.ln = nn.LayerNorm(self.hidden_size)

    def forward(self, x, hidden, encoder_output):
        bs = x.size(0)
        ## hidden state활용
        # print("input size: ", x.size()) # 32, 1
        # print("encoder output: ", encoder_output.size()) # 32, 14, 64
        # print("hidden[0] size: ", hidden[0].size()) # 1, 32, 64
        # print(self.encoder_weight(encoder_output).size())
        # print(self.decoder_weight(hidden[0].permute(1, 0, 2)).size())
        # print((self.encoder_weight(encoder_output) + self.decoder_weight(hidden[0].permute(1, 0, 2))).size() )
        attn_scores = self.value_weight(
            torch.tanh(
                    self.encoder_weight(encoder_output) + 
                    self.decoder_weight(hidden[0].permute(1, 0, 2))
            )
        ).squeeze(2) 
        # print(attn_scores.size()) # 32, 14
        attn_weight = torch.softmax(attn_scores, dim=1)

        # print(attn_weight.permute(0, 2, 1).size())
        # print(attn_weight.size())
        context_vector = torch.bmm(attn_weight.unsqueeze(1), encoder_output).squeeze(1)
        # print("CV : ", context_vector.size()) # 32, 64

        ## Calculate Similiarity scores between context_vector and vectorized events!
        sim_scores = self.gate(
                    torch.matmul(context_vector, self.vectorized_events_mat.permute(1,0))
        ) # 32, 10

        # And concatenate them!
        new_input = torch.cat((context_vector, sim_scores, x), dim=1).unsqueeze(-1) 
        new_input = new_input.permute(0, 2, 1)
        # print("new input size: ", new_input.size()) # 32, 1, 75

        # _, hidden = self.lstm(new_input, hidden) 
        lstm_output, hidden = self.lstm(new_input, hidden) 
        lstm_output = self.ln(lstm_output)
        lstm_output = lstm_output.permute(1,0,2)[0]
        # print(output.size()) # 32, 75, 64
        # print("output Size : ", output.size()) # 32, 64
        # print("hidden Size : ", hidden[0].size()) # 1, 32, 64
        # print("Last Hidden : ", hidden[0][-1].size()) # 32, 64 오!! 이거네

        fin_output = self.fin_linear(lstm_output)
        # print("Final Ouptut Size : ", fin_output.size()) # 32, 1

        return fin_output, hidden, attn_weight, sim_scores


class EventAttentionSeq2SeqModel(nn.Module):
    def __init__(self, vectorized_events_mat, hidden_size):
        super(EventAttentionSeq2SeqModel, self).__init__()
        
        self.hidden_size = hidden_size

        self.encoder = Encoder(input_size=1, hidden_size=hidden_size)
        self.decoder = Decoder(vectorized_events_mat=vectorized_events_mat, input_size=75, hidden_size=hidden_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, inputs, target_len):  # X  # OW
        bs = inputs.shape[0]
        input_size = inputs.shape[2]

        outputs = torch.zeros(bs, target_len, input_size)

        encoder_output, hidden_state = self.encoder(inputs)

        decoder_input = inputs[:, -1, :]  # 최초 Decoder Input

        de_hidden = hidden_state

        total_atten_weight = torch.zeros(bs, 14).to(self.device)
        ## Decoder (예상값 출력)
        for t in range(target_len):  # OW=7이므로 7개의 out을 뱉습니다.
            ## 학습할 땐 sim_scores 확인 안 하는 걸로...
            output, de_hidden, attn_weight, _ = self.decoder(decoder_input, hidden=de_hidden, encoder_output=encoder_output)

            # output = output.squeeze(1)

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

        total_attn_weight = torch.zeros(bs, 14).to(self.device)
        total_sim_scores = torch.zeros(bs, 10).to(self.device)

        for t in range(target_len):
            output, de_hidden, attn_weight, sim_scores = self.decoder(decoder_input, hidden=de_hidden, encoder_output=encoder_output)

            # output = output.squeeze(1)

            decoder_input = output

            outputs[:, t, :] = output
            
            total_attn_weight += attn_weight
            total_sim_scores += sim_scores

        return outputs.detach().numpy()[0, :, 0], total_attn_weight, total_sim_scores

