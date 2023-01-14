from dataloader import *
from dataset import *
from model import *
from prep import *
from utils import *
from trainer import *

import torch
import torch.nn as nn
import torch.optim as optim

## Random Seed를 고정합니다.
seed_everything(seed=43)

# 전처리, 세트화가 완료된 데이터로더를 만듭니다.
data_loader = return_dataloaders()

EPOCHS = 5000
LR = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"

model = NormalSeq2SeqModel(input_size=1, hidden_size=64).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

train = Trainer(
            model=model,
            loader=data_loader,
            epoches=EPOCHS,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

train.doit()


"""
으아아아아 일단 오카방에 물어보자..!
어느 부분에서 Grad가 0이 되는지 한번 체크해보기 
아니면 LSTM 의 output이 nan이 나오는 거면 내부의 grad가 nan이었을 가능성이 높다..!
"""


