from sklearn.preprocessing import MinMaxScaler


def return_prep_data(ori_df):
    tr_df = ori_df[:-7, :]  # 훈련용으로 사용할 데이터들
    te_df = ori_df[-7:, :]  # 예측할 데이터의 정답값

    mm = MinMaxScaler()
    tr_df["Scaled"] = mm.fit_transform(tr_df["original"].values.reshape(-1, 1))
    te_df["Scaled"] = mm.transform(te_df["original"].values.reshape(-1, 1))

    return tr_df, te_df
