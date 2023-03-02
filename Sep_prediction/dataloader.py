import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from dataset import NormalSeq2SeqDataset


def return_dataloaders(model="e"):
    df = pd.read_csv("./data/til_aug_rate.csv")
    if model == "e":        
        Dataset = NormalSeq2SeqDataset(df=df, IW=14, OW=7, stride=1, model="e")
        loader = DataLoader(Dataset, batch_size=32)

        return loader,  Dataset.event_lst, Dataset.fitted_mm
    else:
        Dataset = NormalSeq2SeqDataset(df=df, IW=14, OW=7, stride=1, model="a")
        loader = DataLoader(Dataset, batch_size=32)

        return loader, Dataset.fitted_ss
