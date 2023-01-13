from sklearn.preprocessing import MinMaxScaler


def return_prep_data(ori_df):
    print("📚 전처리를 수행합니다..")

    mm = MinMaxScaler()
    ori_df["rate"] = mm.fit_transform(ori_df["rate"].values.reshape(-1, 1))

    print("✅ Done!")
    return ori_df, mm
