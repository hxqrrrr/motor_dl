import h5py
import pandas as pd
import matplotlib.pyplot as plt
from models.Res_SA import Res_SA
from models.Res_SA_new import *
def plt_data(numpy):
    df = pd.DataFrame(numpy)
    for i in range(df.shape[0]):
        plt.plot(df.iloc[i])
    plt.show()
with h5py.File('data.h5', 'r') as f:
    # 读取数据
    data = f['data'][:]
    # 读取标签
    labels = f['labels'][:]
    plt_data(data[1])
    print(data[4])
   


