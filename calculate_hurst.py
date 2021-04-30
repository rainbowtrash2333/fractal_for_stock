import matplotlib.pyplot as plt
import pandas as pd
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import sklearn.preprocessing
import multiprocessing as mp
import time
from pylab import mpl
import hurst


def run(ts):
    H, c, data = hurst.compute_Hc(ts)
    return H


def make_hurst_list_pool(data, ts_len):
    pool = mp.Pool()
    ts = []
    for i in range(len(data) - ts_len + 1):
        ts.append(data[i:i + ts_len])
    time_start = time.time()
    result = [pool.apply_async(run, args=(i,)) for i in ts]
    pool.close()
    pool.join()
    hurst_list = [i.get() for i in result]
    # hurst_list = np.asanyarray(hurst_list)
    time_end = time.time()
    print('hurst计算耗时', time_end - time_start, 's')
    for i in range(ts_len - 1):
        hurst_list.insert(i, hurst_list[ts_len - 1])
    #   np.insert(hurst_list, i, hurst_list[ts_len - 1])
    return hurst_list


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
    mpl.rcParams['axes.unicode_minus'] = False
    # dataset = pd.read_csv(r'D:\tensorflow-gpu\code\hpq.us.txt')
    dataset = pd.read_csv(r'D:\毕业设计\Data\Stocks\hpq.us.txt')
    df = dataset.sort_values('Date')

    # 计算中间价格
    high_prices = df.loc[:, 'High'].to_numpy()
    low_prices = df.loc[:, 'Low'].to_numpy()
    mid_prices = (high_prices + low_prices) / 2.0
    train_data = mid_prices[:11000]
    test_data = mid_prices[11000:]
    # 将最高价与最低价求均值，得到当天的均价
    scaler = sklearn.preprocessing.MinMaxScaler()
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)
    smoothing_window_size = 2500
    for di in range(0, 10000, smoothing_window_size):
        scaler.fit(train_data[di:di + smoothing_window_size, :])
        train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])

    scaler.fit(train_data[di + smoothing_window_size:, :])
    train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])
    train_data = train_data.reshape(-1)
    test_data = scaler.transform(test_data).reshape(-1)
    EMA = 0.0
    gamma = 0.1
    for ti in range(400):
        EMA = gamma * train_data[ti] + (1 - gamma) * EMA
        train_data[ti] = EMA

    all_mid_data = np.concatenate([train_data, test_data], axis=0)

    ts_len_list = [100, 150, 200, 500, 1000, 3000]
    # ts_len_list = [100, 150]
    Hurst_list = [make_hurst_list_pool(all_mid_data, ts_len) for ts_len in ts_len_list]

    data_transposed = zip(Hurst_list[0], Hurst_list[1], Hurst_list[2], Hurst_list[3], Hurst_list[4], Hurst_list[5])
    df = pd.DataFrame(data_transposed, columns=ts_len_list)

    df.to_csv(r'.\Data\myData\hurst.csv', index=False)
