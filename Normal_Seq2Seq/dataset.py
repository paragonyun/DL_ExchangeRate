import pandas as pd
import numpy as np

from torch.utils.data import Dataset

from prep import return_prep_data

IW = 14  # 이전 2주치를 보고
OW = 7  # 일주일을 예측


class NormalSeq2SeqDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        return ...

    def __len__(self):
        return len()
