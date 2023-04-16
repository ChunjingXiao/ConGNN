import numpy as np
import pandas as pd


def multiplication_thred():
    dir_path = 'experiments/SMD_TEST_128_2048_10_100/results/'
    data_path = dir_path + 'all.csv'
    thred = 0.20059595
    ori_df = pd.read_csv(data_path)
    df = ori_df.copy()
    small = df[df['differ'] < thred].index.tolist()

    for i in small:
        df.loc[i, 'differ'] = df.loc[i, 'differ'] / 1.0

    df.columns = ['0', '1', '2', '3']

    df.drop(columns=['3'], inplace=True)
    df.to_csv(dir_path + "smd" + "_my_test.csv", index=False)


if __name__ == '__main__':
    multiplication_thred()
