from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch

def return_prep_data(ori_df):
    print("📚 전처리를 수행합니다..")
    print("차분 수행중...")
    diff_rate = ori_df['rate'].diff(1)
    ori_df = ori_df.iloc[1:, :]
    ori_df['rate'] = diff_rate
    first = pd.DataFrame({"Time": ["1997/12/01"],
                        "rate": [0]})
    ori_df = pd.concat([first, ori_df], axis=0)
    copy_df = ori_df.copy()

    print("Scaling 진행중...")
    ss = StandardScaler()
    copy_df["rate"] = ss.fit_transform(copy_df["rate"].values.reshape(-1, 1))

    print("✅ Done!")
    return ori_df, ss
