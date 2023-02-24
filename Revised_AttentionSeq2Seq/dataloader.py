import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from dataset import NormalSeq2SeqDataset


def return_dataloaders():
    df = pd.read_csv("./data/revised_exchange_rate.csv")
    Dataset = NormalSeq2SeqDataset(df=df, IW=60, OW=20, stride=1)
    loader = DataLoader(Dataset, batch_size=32)

    return loader,  Dataset.event_lst, Dataset.fitted_mm
