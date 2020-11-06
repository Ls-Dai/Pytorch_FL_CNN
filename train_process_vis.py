import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from configs import TrainConfig

if __name__ == '__main__':
    config = TrainConfig()
    file_path_list = []
    for i in range(config.num_of_clients):
        file_path = "clients/" + str(i) + "/log.csv"
        file_path_list.append(file_path)

    client_curve_list = []
    plt.figure()
    for count, file_path in enumerate(file_path_list):
        df = pd.read_csv(file_path, header=None)
        # print(df)
        arr = np.array(df)
        # print(arr)
        lst = arr.tolist()
        # print(lst)
        loss_curve = []
        for l in lst:
            loss_curve.append(l[1])
        plt.plot(loss_curve)
        client_curve_list.append('client_' + str(count))
    plt.legend(client_curve_list)
    plt.show()
