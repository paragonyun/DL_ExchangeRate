import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader
from dataset import NormalSeq2SeqDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def return_dataloaders(model="e"):
    df = pd.read_csv("./data/til_aug_rate.csv")
    train_df = df.iloc[:-42, :]
    val_df = df.iloc[-42:-21, :]
    test_df = df.iloc[-21:, :]


    if model == "e":        
        Train_Dataset = NormalSeq2SeqDataset(df=train_df, IW=14, OW=7, stride=1, model="e")
        loader = DataLoader(Train_Dataset, batch_size=32)

        validation_x, validation_y = _make_set(val_df, Train_Dataset.fitted_scaler)
        test_x, test_y = _make_set(test_df, Train_Dataset.fitted_scaler)

        return loader,  Train_Dataset.event_lst, (validation_x, validation_y), (test_x, test_y)
        # return loader,  Dataset.event_lst
    else:
        Train_Dataset = NormalSeq2SeqDataset(df=train_df, IW=14, OW=7, stride=1, model="a")
        loader = DataLoader(Train_Dataset, batch_size=32)

        validation_x, validation_y = _make_set(val_df, Train_Dataset.fitted_scaler)
        test_x, test_y = _make_set(test_df, Train_Dataset.fitted_scaler)

        return loader, (validation_x, validation_y), (test_x, test_y)
        # return loader



def _make_set(ori_df, fitted_scaler):

    features = ori_df.iloc[:14, :]
    labels = ori_df.iloc[14:, :]

    diffs = features["rate"].diff()
    scaled = fitted_scaler.transform(diffs.reshape(-1,1))
    for_preds = torch.tensor(scaled).to(device).float()
    labels = torch.tensor(labels).to(device).float()

    return for_preds, labels