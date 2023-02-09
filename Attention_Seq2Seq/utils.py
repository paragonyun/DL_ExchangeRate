import torch
import random
import numpy as np
import os

import matplotlib.pyplot as plt


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def plot_result(ori_df, preds):
    plt.figure(figsize=(20, 5))

    plt.plot(range(6000, 6284), ori_df["rate"][6000:], label="Actual")

    plt.plot(range(6290 - 7, 6290), preds, label="predict")

    plt.savefig('./Results.png')
    plt.legend()
    plt.show()

def change_to_original(df, preds):
    fin_value = df['rate'].iloc[-1]
    # print(fin_value)
    fin_preds = [fin_value]
    for diff in preds:
        fin_value += diff
        fin_preds.append(fin_value)
    fin_preds.pop(0)
    
    return fin_preds
