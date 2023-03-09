import pandas as pd
import numpy as np

from torch.utils.data import Dataset

from prep import return_prep_data


class NormalSeq2SeqDataset(Dataset):
    """Idea
    2주치를 보고 1주일을 예측하는 모델을 개발하기 위해선
    Input Window와 output Window를 설정하여 해당 기간들이 담긴 데이터셋을 생성해야합니다.

    ex) [1,2,3,4,5,6,7,8,9,10] , IW = 3, OW = 1, Stride = 1
    Inputs -> [1,2,3] , [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]
    Outputs -> [4], [5], [6], [7], [8], [9], [10]
    num_samples = Lenth - IW - OW // (stride + 1)
                = ((10 - 3 - 1) // 1 ) + 1
                = 6//1 + 1 = 7개
    """

    def __init__(self, df, IW=14, OW=7, stride=1, model="e"):
        # 전처리
        if model == "e":
            prep_df, self.event_lst, self.fitted_mm = return_prep_data(df, model="e")
        else:
            prep_df, self.fitted_ss = return_prep_data(df, model="a")
        # if model == "e":
        #     prep_df, self.event_lst = return_prep_data(df, model="e")
        # else:
        #     prep_df = return_prep_data(df, model="a")

        Length = df.shape[0]

        num_samples = ((Length - IW - OW) // stride) + 1

        Xs = np.zeros([IW, num_samples])
        Ys = np.zeros([OW, num_samples])

        for i in range(num_samples):  # Samples를 만듭니다.
            x_start = stride * i
            x_end = x_start + IW
            Xs[:, i] = prep_df["rate"][x_start:x_end]

            y_start = stride * i + IW
            y_end = y_start + OW

            Ys[:, i] = prep_df["rate"][y_start:y_end]

        Xs = Xs.reshape(Xs.shape[0], Xs.shape[1], 1).transpose((1, 0, 2))
        Ys = Ys.reshape(Ys.shape[0], Ys.shape[1], 1).transpose((1, 0, 2))

        self.Xs = Xs
        self.Ys = Ys

    def __getitem__(self, idx):
        return self.Xs[idx], self.Ys[idx]

    def __len__(self):
        return len(self.Xs)
