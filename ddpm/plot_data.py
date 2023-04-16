import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_anomaly(file_csv):
    data_name = file_csv.split('/')[-1].split('_')[0]
    label_path = file_csv.replace('test', 'test_label')
    # Read the raw data.
    data = pd.read_csv(file_csv)
    labels = pd.read_csv(label_path)

    col_num = data.shape[1]
    col_names = []
    for i in range(col_num):
        col_names.append('value_' + str(i))
    data.columns = col_names

    if data_name.upper() != 'PSM':
        labels.rename(columns={'0': 'label'}, inplace=True)

    # timestamp = data["timestamp"]

    ORI = pd.DataFrame()

    num = 0
    col_name = 'value_' + str(num)
    ORI['value'] = data[col_name]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # psm 4 3200-3500
    start = 437000
    end = 440000
    ax.plot(ORI[start:end])
    ano_values = []
    for i in range(start, end):
        if labels.loc[i, 'label'] == 0:
            ano_values.append(None)
        else:
            ano_values.append(ORI.loc[i, 'value'])
    ax.plot(range(start, end), ano_values, c="r")

    ax.set_title(data_name + "_" + str(num))

    plt.tight_layout()
    plt.show()


def plot(file_csv):
    # Read the raw data.
    data = pd.read_csv(file_csv)

    SR = data["SR"]
    INF = data["INF"]
    ORI = data["value"]
    DROP = data["DROP"]
    fig = plt.figure()
    ax = fig.add_subplot(4, 1, 1)
    bx = fig.add_subplot(4, 1, 2)
    cx = fig.add_subplot(4, 1, 3)
    dx = fig.add_subplot(4, 1, 4)
    start = 0
    end = 128
    ax.plot(SR[start:])
    bx.plot(INF[start:])
    cx.plot(ORI[start:])
    dx.plot(DROP[start:])
    ax.set_title("SR")
    bx.set_title("INF")
    cx.set_title("ORI")
    dx.set_title("DROP")

    plt.tight_layout()
    plt.show()


def drop(file_csv):
    # Read the raw data.
    data = pd.read_csv(file_csv)
    SR = data["SR"]
    ORI = data["ORI"]
    x = range(0, 128)

    y = ORI.copy()

    z = np.polyfit(x, y, 10)
    fx = np.poly1d(z)
    DROPS = []
    for i in x:
        gen = fx(i)
        DROPS.append(gen)

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    bx = fig.add_subplot(2, 1, 2)

    start = 0
    end = 128
    ax.plot(DROPS[start:])
    bx.plot(SR[start:])

    ax.set_title("DROP")
    bx.set_title("ORI")
    plt.tight_layout()
    plt.show()


def insert(file_csv):
    # Read the raw data.
    data = pd.read_csv(file_csv)
    SR = data["SR"]
    ORI = data["ORI"]
    r = int(ORI.__len__() / 8)
    x = np.arange(0, r, 1 / 8)

    y = ORI.copy()
    fp = []
    for i in range(0, ORI.__len__(), 8):
        fp.append(y[i])
    xp = range(0, r)
    z = np.interp(x, xp, fp)

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    bx = fig.add_subplot(2, 1, 2)
    start = 0
    end = 128
    ax.plot(z[start:])
    bx.plot(ORI[start:])

    ax.set_title("INSERT")

    bx.set_title("ORI")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    data_name = 'smd'
    file_csv = "tf_dataset/" + data_name + "/" + data_name + "_test.csv"

    plot_anomaly(file_csv)

