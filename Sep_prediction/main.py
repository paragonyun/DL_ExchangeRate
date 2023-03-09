from dataloader import *
from dataset import *
from model_attn import *
from model_event_attn import *
from model_normal import *
from prep import *
from utils import *
from trainer import *

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", dest="model")
args = parser.parse_args()

## Random Seed를 고정합니다.
seed_everything(seed=43)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"현재 모드 : {args.model}")
# 전처리, 세트화가 완료된 데이터로더를 만듭니다.
if args.model == "e":
    data_loader, events_mat, fitted_ss = return_dataloaders(model=args.model)
    with torch.no_grad():
        vectorizer = nn.Linear(in_features=7, out_features=64, bias=False)
        nn.init.xavier_uniform_(vectorizer.weight)
        vectorized_events = torch.tensor(vectorizer(events_mat).detach().numpy()).to(device)

else:
    data_loader, fitted_ss = return_dataloaders(model=args.model)
# if args.model == "e":
#     data_loader, events_mat= return_dataloaders(model=args.model)
#     with torch.no_grad():
#         vectorizer = nn.Linear(in_features=7, out_features=64, bias=False)
#         nn.init.xavier_uniform_(vectorizer.weight)
#         vectorized_events = torch.tensor(vectorizer(events_mat).detach().numpy()).to(device)

# else:
#     data_loader = return_dataloaders(model=args.model)

EPOCHS = 1
LR = 0.001


if args.model == "e":
    model = EventAttentionSeq2SeqModel(vectorized_events_mat=vectorized_events,  hidden_size=64).to(device)
elif args.model == "a":
    model = AttentionSeq2SeqModel( hidden_size=64).to(device) 
elif args.model == "n":
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
    version=args.model
)
if args.model == "a" or args.model == "e":
    atten_weights = train.doit() 
else:
    train.doit()

## Evaluation
print("✨Start Evaluation...✨")

ori_df = pd.read_csv("./data/til_aug_rate.csv")
diffs = ori_df["rate"].diff()
scaled = fitted_ss.transform(diffs.iloc[-14:].values.reshape(-1, 1))
input_data = torch.tensor(scaled).to(device).float()

# input_data = torch.tensor(diffs.iloc[-14:].values.reshape(-1, 1)).to(device).float()

if args.model == "e":
    model.load_state_dict(torch.load("./EVENT_BEST_MODEL.pth"))
    predict, atten_weights, sim_scores = model.predict(inputs=input_data, target_len=7)

elif args.model == "a":
    model.load_state_dict(torch.load("./ATTENTION_BEST_MODEL.pth"))
    predict, atten_weights = model.predict(inputs=input_data, target_len=7)

elif args.model == "n":
    model.load_state_dict(torch.load("./NORMAL_BEST_MODEL.pth"))
    predict = model.predict(inputs=input_data, target_len=7)


actuals = ori_df["rate"].to_numpy()
print("Original Prediction (Before Inverse Transform)")
print(predict, "\n")

print("Inverse Trasforming...")
predict = fitted_ss.inverse_transform(predict.reshape(-1, 1))
actuals = fitted_ss.inverse_transform(actuals.reshape(-1, 1))
print("Inversed Changes\n", predict)

predictions = change_to_original(df=ori_df, preds=predict)

if args.model == "e" or args.model == "a":
    print("\nAttention Weights")
    print(atten_weights,"\n")

if args.model == "e":
    print("\nEvent Similarity (Check the prep.py's events)")
    print(sim_scores, "\n\n")

# print("👀Prediction👀")
# print(predictions)

# print("Plot Results...")
# plot_result(ori_df=ori_df, preds=predictions)
