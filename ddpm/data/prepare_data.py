import math

import numpy as np
import pandas as pd
import torch
import random

import core.metrics as Metrics
import warnings

warnings.filterwarnings("ignore")


class PrepareData:
    def __init__(self, data_path, phase, base, size):
        self.data_path = data_path
        self.phase = phase
        self.base = base
        self.size = size

        self.data_name = self.data_path.split('/')[-1].split('_')[0]
        self.read_dataset(self.data_path, self.data_name)
        self.df = self.ori_df.copy()
        self.row_num = self.ori_df.shape[0]
        self.col_num = self.ori_df.shape[1]
        self.mean = self.df.mean(axis=1)
        self.df = self.fill_data(self.df)


    def get_hr_data(self, base, size):
        df = self.df.copy()
        ori_values, values, labels, pre_labels = self.get_data_by_interval(df)

        return ori_values, values, labels, pre_labels

    def get_sr_data(self):
        df = self.df.copy()

        ori_values, values, labels, pre_labels = self.get_data_by_interval(df)

        return ori_values, values, labels, pre_labels

    def get_mean_df(self, df):
        df = df.copy()
        for col in df.columns:
            df[col] = self.mean
        return df

    def vertical_merge_df(self, df):
        df = df.copy()
        two_power = 2

        if self.col_num == 1:
            two_power = 16

            df_temp = pd.DataFrame()
            for i in range(two_power - self.col_num):
                df_temp[i] = df.iloc[:, 0]

        else:
            while self.col_num > two_power:
                two_power = two_power * 2
            df_temp = df.iloc[:, 0:(two_power - self.col_num)]

        col_name = []
        for i in range(self.col_num):
            col_name.append('value_' + str(i))

        df.columns = col_name

        col_name = []
        for i in range(self.col_num, two_power):
            col_name.append('value_' + str(i))

        df_temp.columns = col_name

        df = pd.concat([df, df_temp], axis=1)
        return df


    def join_together_labels(self, df):
        df = df.copy()

        if self.phase == 'train':
            df['label'] = 0
        else:
            df['label'] = self.test_labels
        return df


    def fill_data(self, df):
        df = df.copy()
        data_end = math.ceil(self.row_num / 128) * 128

        df = np.pad(df,((0,data_end-self.row_num,),(0,0)),'constant')
        
        return df


    def read_dataset(self, data_path, data_name):
            self.get_dataset(data_path)

    def get_dataset(self, data_path):
        if self.phase == 'train':

            self.ori_df = np.load(self.data_path,allow_pickle=True)
        else:

            self.ori_df = np.load(self.data_path,allow_pickle=True)


    def get_data_by_interval(self, df):

        data = np.load(self.data_path,allow_pickle=True)
        data_end = math.ceil(self.row_num / 128) * 128
        mm_data = np.pad(data,((0,data_end-self.row_num),(0,0)),'constant')
        mm_data = np.expand_dims (mm_data,axis=0)

        # mm_data = np.tile(mm_data,(1,1,1))
        mm_data = np.split(mm_data,self.row_num//128+1,axis=1)
        values = []
        for data in  mm_data:
            values.append(torch.tensor(data.astype(np.float32)))

        return values, values, None, None

    def complete_value(self, df):
        df.fillna(0, inplace=True)
        return df

    def get_row_num(self):
        return self.row_num

    def get_col_num(self):
        return self.col_num


if __name__ == '__main__':
    data_path = ""
    base = 128
    size = 128
    pre = PrepareData(data_path,"train",base,size)
    pre.get_hr_data( base, size)
