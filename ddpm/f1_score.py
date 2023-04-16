import pandas as pd

import core.metrics as Metrics
from decimal import Decimal

if __name__ == '__main__':

    start_label = 2000
    end_label = 2200
    step_label = 1
    step_t = 100

    start_epoch = 100
    end_epoch = 100
    step_epoch = 1000

    epochs = range(start_epoch, end_epoch + 1, step_epoch)


    params = {
        'start_label': start_label,
        'end_label': end_label,
        'step_label': step_label,
        'step_t': step_t
    }

    temp_list = []

    for epoch in epochs:
        file_name = "swat1_all.csv"
        folder_name = "SMAP_TEST_128_2048_100_" + str(epoch) + "/results/"
        file_csv = "experiments/" + folder_name + file_name

        # Read the raw data.
        df = pd.read_csv(file_csv)

        mean_dict = Metrics.get_val_mean(df)
        best_N, best_thred, accuracy, precision, recall, best_f1 = Metrics.relabeling_strategy(df, params)

        temp_f1 = Decimal(best_f1 * 100).quantize(Decimal("0.00"))
        temp_MSE = Decimal(mean_dict['MSE'].astype(Decimal)).quantize(Decimal("0.0000"))
        temp_map = {'F1': float(temp_f1), 'MSE': float(temp_MSE), 'best_thred': best_thred,
                    'model_epoch': epoch, 'N': best_N}
        temp_list.append(temp_map)

        # best_threshold, accuracy, precision, recall, best_f1
        print("==================" + str(epoch) + "======================")
        print("best_thred: ", best_thred)
        print("accuracy: ", accuracy)
        print("precision: ", precision)
        print("recall: ", recall)
        print("best_f1: ", best_f1)

    print(temp_list)
