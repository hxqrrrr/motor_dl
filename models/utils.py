import pandas as pd
import matplotlib.pyplot as plt

def plt_data(numpy):
    df = pd.DataFrame(numpy)
    for i in range(df.shape[0]):
        plt.plot(df.iloc[i])
    plt.show()