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
data_loader, fitted_ss = return_dataloaders()

EPOCHS = 3000
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
    device=device,
)

train.doit()

## Evaluation
print("✨Start Evaluation...✨")

ori_df = pd.read_csv("./data/for_oct_revised_pred.csv")
diffs = ori_df["rate"].diff()
scaled = fitted_ss.transform(diffs.iloc[-14:].values.reshape(-1, 1))
input_data = torch.tensor(scaled).to(device).float()

model.load_state_dict(torch.load("./BEST_MODEL.pth"))

predict = model.predict(inputs=input_data, target_len=20)

actuals = ori_df["rate"].to_numpy()
print("Original Prediction (Before Inverse Transform)")
print(predict, "\n")

print("Inverse Trasforming...")
predict = fitted_ss.inverse_transform(predict.reshape(-1, 1))
actuals = fitted_ss.inverse_transform(actuals.reshape(-1, 1))
print("Inversed Changes\n", predict)

predictions = change_to_original(df=ori_df, preds=predict)

print("👀Prediction👀")
print(predictions)

print("Plot Results...")
plot_result(ori_df=ori_df, preds=predictions)

## 23년 1월 예측 - input : 22년 10,11,12월 
print("✨Start Evaluation...✨")

ori_df = pd.read_csv("./data/for_23y_revised_rate.csv")
diffs = ori_df["rate"].diff()
scaled = fitted_ss.transform(diffs.iloc[-14:].values.reshape(-1, 1))
input_data = torch.tensor(scaled).to(device).float()

model.load_state_dict(torch.load("./BEST_MODEL.pth"))

predict = model.predict(inputs=input_data, target_len=20)

actuals = ori_df["rate"].to_numpy()
print("Original Prediction (Before Inverse Transform)")
print(predict, "\n")

print("Inverse Trasforming...")
predict = fitted_ss.inverse_transform(predict.reshape(-1, 1))
actuals = fitted_ss.inverse_transform(actuals.reshape(-1, 1))
print("Inversed Changes\n", predict)

predictions = change_to_original(df=ori_df, preds=predict)

print("👀Prediction👀")
print(predictions)

print("Plot Results...")
plot_result(ori_df=ori_df, preds=predictions)
