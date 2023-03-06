from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch

def return_prep_data(ori_df, model="e"):
    print("📚 전처리를 수행합니다..")
    print("차분 수행중...")
    diff_rate = ori_df['rate'].diff(1)
    ori_df = ori_df.iloc[1:, :]
    ori_df['rate'] = diff_rate
    first = pd.DataFrame({"Time": ["1997/12/01"],
                        "rate": [0]})
    ori_df = pd.concat([first, ori_df], axis=0)
    copy_df = ori_df.copy()

    # print("Scaling 진행중...")
    # ss = StandardScaler()
    # copy_df["rate"] = ss.fit_transform(copy_df["rate"].values.reshape(-1, 1))

    if model == "e":
        print("Impact Event를 찾는 중...")
        times = ["1998/11/24", "1998/01/03", "2020/12/07", "2008/10/08", "2008/08/27", 
                "2013/04/12", "2010/06/30", "2008/04/22", "2014/03/07", "2014/08/21" ]
        event_tensor = find_impact_events(ori_df=ori_df, times=times)
        
        print("✅ Done!")
        # return copy_df, event_tensor, ss
        return copy_df, event_tensor
    
    else:
        print("✅ Done!")
        # return ori_df, ss
        return ori_df


def find_impact_events(ori_df, times: list):
    # time -> "year/month/day"
    event_arr = np.array([])

    for time in times:
        idx = ori_df[ori_df['Time'] == time].index[0]
        values = np.array(ori_df.iloc[idx-4 : idx + 3, 1])
        event_arr = np.concatenate((event_arr, values), axis=0)
    
    event_arr = torch.tensor(event_arr.reshape(len(times), 7), dtype=torch.float32)

    return event_arr
