#Signal to noise ratio
import numpy as np
import pandas as pd
import os.path as path

data = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "Data_processing", "normalized_test.csv"))

def signaltonoise(a, axis = 0, ddof = 0): 
    a = np.asanyarray(a) 
    m = a.mean(axis) 
    sd = a.std(axis = axis, ddof = ddof) 
    return np.where(sd == 0, 0, m / sd) 

arr = data['A']
snr = signaltonoise(arr)
print(snr)