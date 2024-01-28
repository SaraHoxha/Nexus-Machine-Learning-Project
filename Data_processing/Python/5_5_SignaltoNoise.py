#Signal to noise ratio
import numpy as np
import scipy.io
import pandas as pd

data = pd.read_csv("C:/Users/urbi1/OneDrive/Escritorio/ML_2023/NN/normalized_training.csv")

def signaltonoise(a, axis = 0, ddof = 0): 
    a = np.asanyarray(a) 
    m = a.mean(axis) 
    sd = a.std(axis = axis, ddof = ddof) 
    return np.where(sd == 0, 0, m / sd) 

arr = data['A']
snr = signaltonoise(arr)
print(snr)