import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, mean_squared_error


# 将tensor数据的维度为1的压缩
def squeeze_tensor(tensor):
    return tensor.squeeze().cpu()


# 更新列名
def update_csv_col_name(all_datas):
    df = all_datas.copy()
    df.columns = [0, 1, 2, 3]

    return df


# 将sr、hr、lr、inf全都写入一个csv文件中
def tensor2allcsv(visuals, col_num):
    df = pd.DataFrame()
    # 将三维的SR数据压缩为二维
    sr_df = pd.DataFrame(squeeze_tensor(visuals['SR']))
    # 将四维的ORI数据压缩为二维
    ori_df = pd.DataFrame(squeeze_tensor(visuals['ORI']))
    lr_df = pd.DataFrame(squeeze_tensor(visuals['LR']))
    # 存储各列的差值
    differ_df = pd.DataFrame()

    # 如果真实列数为1，则不删除填充的列，便于寻找合适图像
    if col_num != 1:
        # 删除填充的列
        for i in range(col_num, sr_df.shape[1]):
            sr_df.drop(labels=i, axis=1, inplace=True)
            ori_df.drop(labels=i, axis=1, inplace=True)
            lr_df.drop(labels=i, axis=1, inplace=True)

    df['SR'] = sr_df.mean(axis=1)
    df['ORI'] = ori_df.mean(axis=1)
    df['LR'] = lr_df.mean(axis=1)
    # 每行的原始值和生成值之间的平均差
    # df['differ'] = (ori_df - sr_df).abs().mean(axis=1)
    df['differ'] = (ori_df - sr_df).abs().mean(axis=1)
    df['label'] = squeeze_tensor(visuals['label'])

    # TODO 画图临时修改
    # for i in range(sr_df.shape[1]):
    #     differ_df[str(i)] = (ori_df - sr_df).abs()
    #     # differ_df[str(i)] = ori_df[i] - sr_df[i]
    # differ_df = (ori_df - sr_df).abs()
    # differ_df = (ori_df - sr_df)
    differ_df = (sr_df - ori_df)

    return df, sr_df, differ_df


# 将分块保存的所有数据存入all_datas
def merge_all_csv(all_datas, all_data):
    all_datas = pd.concat([all_datas, all_data])
    return all_datas


def save_csv(data, data_path):
    data.to_csv(data_path, index=False)


# 获得正常点、异常点、全部点（包括异常点和正常点）的平均值
def get_mean(df):
    # 获得全部点的平均值
    mean = df['value'].astype('float32').mean()
    # 获得正常点的平均值
    normal_mean = df['value'][df['label'] == 0].astype('float32').mean()
    # 获得异常点的平均值
    anomaly_mean = df['value'][df['label'] == 1].astype('float32').mean()

    return mean, normal_mean, anomaly_mean


# 获得验证集生成正常点、生成异常点点、原始正常点、原始异常点、生成点（包括异常点和正常点）、原始点（包括异常点和正常点）的平均值
def get_val_mean(df):
    mean_dict = {}
    # TODO 原始为ori
    ori = 'ORI'
    ori_mean = df[ori].astype('float32').mean()
    # 获得原始的正常点的平均值
    ori_normal_mean = df[ori][df['label'] == 0].astype('float32').mean()
    # 获得原始的异常点的平均值
    ori_anomaly_mean = df[ori][df['label'] == 1].astype('float32').mean()

    gen_mean = df['SR'].astype('float32').mean()
    # 获得生成的正常点的平均值
    gen_normal_mean = df['SR'][df['label'] == 0].astype('float32').mean()
    # 获得生成的异常点的平均值
    gen_anomaly_mean = df['SR'][df['label'] == 1].astype('float32').mean()

    # 计算均方误差
    mean_dict['MSE'] = mean_squared_error(df[ori], df['SR'])

    mean_dict['ori_mean'] = ori_mean
    mean_dict['ori_normal_mean'] = ori_normal_mean
    mean_dict['ori_anomaly_mean'] = ori_anomaly_mean

    mean_dict['gen_mean'] = gen_mean
    mean_dict['gen_normal_mean'] = gen_normal_mean
    mean_dict['gen_anomaly_mean'] = gen_anomaly_mean

    # 原始均值和生成均值的差
    mean_dict['mean_differ'] = ori_mean - gen_mean
    # 正常点的均值差
    mean_dict['normal_mean_differ'] = ori_normal_mean - gen_normal_mean
    # 异常点的均值差
    mean_dict['anomaly_mean_differ'] = ori_anomaly_mean - gen_anomaly_mean

    # 原始正常点和异常点的均值差
    mean_dict['ori_no-ano_differ'] = ori_normal_mean - ori_anomaly_mean
    # 原始均值和正常点均值的差
    mean_dict['ori_mean-no_differ'] = ori_mean - ori_normal_mean
    # 原始均值和异常点均值的差
    mean_dict['ori_mean-ano_differ'] = ori_mean - ori_anomaly_mean

    # 生成正常点和异常点的均值差
    mean_dict['gen_no-ano_differ'] = gen_normal_mean - gen_anomaly_mean
    # 生成均值和正常点均值的差
    mean_dict['gen_mean-no_differ'] = gen_mean - gen_normal_mean
    # 生成均值和异常点均值的差
    mean_dict['gen_mean-ano_differ'] = gen_mean - gen_anomaly_mean

    return mean_dict


# 以预测到的真正的异常点为中心，将连续的异常都标记为被预测的异常点
def relabeling_strategy(df, params):
    y_true = []
    best_N = 0
    best_f1 = -1
    best_thred = 0
    best_predictions = []
    # 设置多个阈值，看哪个效果最好
    thresholds = np.arange(params['start_label'], params['end_label'], params['step_label'])

    # 对差值进行降序排序
    df_sort = df.sort_values(by="differ", ascending=False)
    # 重置索引, drop=False代表保留原索引
    df_sort = df_sort.reset_index(drop=False)

    for t in thresholds:
        if (t - 1) % params['step_t'] == 0:
            print("t: ", t)
        y_true, y_pred, thred = predict_labels(df_sort, t)
        # TODO 策略 以预测到的真正的异常点为中心，将连续的异常都标记为被预测的异常点
        for i in range(len(y_true)):
            if y_pred[i] == 1 and y_true[i] == 1:
                # 以i为中心向前一个个判断是否异常
                j = i - 1
                while j >= 0 and y_true[j] == 1 and y_pred[j] == 0:
                    y_pred[j] = 1
                    j -= 1
                # 以i为中心向后一个个判断是否异常
                j = i + 1
                while j < len(y_pred) and y_true[j] == 1 and y_pred[j] == 0:
                    y_pred[j] = 1
                    j += 1

        f1 = calculate_f1(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_N = t
            best_thred = thred
            best_predictions = y_pred

    accuracy = calculate_accuracy(y_true, best_predictions)
    precision = calculate_precision(y_true, best_predictions)
    recall = calculate_recall(y_true, best_predictions)

    return best_N, best_thred, accuracy, precision, recall, best_f1


# 预测数据的标签
def predict_labels(df_sort, num):
    # 排序靠前的数据看作异常点,将预测标签设置为1
    df_sort['pred_label'] = 0
    df_sort.loc[0:num - 1, 'pred_label'] = 1
    thred = df_sort.loc[num - 1, 'differ']

    # 将“index”列设置为索引
    df_sort = df_sort.set_index('index')
    # 按照索引排序
    df_sort = df_sort.sort_index()

    y_true = df_sort['label'].tolist()
    y_pred = df_sort['pred_label'].tolist()

    return y_true, y_pred, thred


# 计算准确率
def calculate_accuracy(y_true, y_pred):
    # accuracy_score()是sk中计算准确率的方法
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


# 计算精确率
def calculate_precision(y_true, y_pred):
    # precision_score()是sk中计算精确率的方法
    precision = precision_score(y_true, y_pred)
    return precision


# 计算召回率
def calculate_recall(y_true, y_pred):
    # recall_score()是sk中计算召回率的方法
    recall = recall_score(y_true, y_pred)
    return recall


# 计算F1分数
def calculate_f1(y_true, y_pred):
    # f1_score()是sk中计算F1分数的方法
    f1 = f1_score(y_true, y_pred)

    return f1
