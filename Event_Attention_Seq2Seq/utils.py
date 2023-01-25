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

"""
plan
1. prep에서  차분 실시
2. 차분하고 제일 높은 차이를 보이는 topn 추출(연도 기반..?)
3. 추출한 일자 쁠마 3일씩 해서 7일치를 얻어냄 -> B x n x 7이 될 거임
4. 그 7일치를 어떻게는 Vetor로 표현 
    - Vector로 표현하는 방법
    (1) 그냥 tensor그 자체로 context vector와 cosine similarity
    (2) tensor를 nn.Linear에 넣어서 B x n x hidden 으로 만든 다음 이 vectors와 context vector를 비교 ** 이걸로 해보자..!
    (3) B x n x 7 의 tensor를 LSTM에 넣어서 B x n x hidden 으로 만든 다음 이 Vectors와 Context Vector 비교 (nn.Linear)
    (4) torch.embedding은 float은 안 되는듯...
"""

import torch.nn as nn
import torch
def Embdder(float_tensors: torch.tensor, num_embeds: int = 8, emb_dim: int = 64):
        emb = nn.Embedding(num_embeddings=num_embeds, embedding_dim=emb_dim)
    
        input_floats = float_tensors.long()
        
        embedded_vectors = emb(input_floats)
        print(embedded_vectors.size())
        print(embedded_vectors)

        return embedded_vectors

arr = torch.tensor([[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                    [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
                    [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7],
                    [4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7],
                    [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7],
])

vectors = Embdder(arr)
