import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

def plt_data(numpy):
    df = pd.DataFrame(numpy)
    for i in range(df.shape[0]):
        plt.plot(df.iloc[i])
    plt.show()

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)