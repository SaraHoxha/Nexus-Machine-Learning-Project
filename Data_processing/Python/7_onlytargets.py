#In this file we exclude the targets to then test against the predicted targets
#This is done in the end of the project when the final models are ready

#Load libraries
import pandas as pd
import numpy as np

#Import the data sets
data = pd.read_csv("C:/Users/urbi1/OneDrive/Escritorio/ML_2023/Data_split/train70.csv")
data_test = pd.read_csv("C:/Users/urbi1/OneDrive/Escritorio/ML_2023/Data_split/test30.csv")

#Slice the data frames to only leave the targets
targets_training = data.iloc[:, -3:]
targets_test = data_test.iloc[:, -3:]

#Print and check the targets
targets_training
targets_test

#Save the targets
targets_training.to_csv('C:/Users/urbi1/OneDrive/Escritorio/ML_2023/Data_split/onlytargets_train70.csv', index=False)
targets_test.to_csv('C:/Users/urbi1/OneDrive/Escritorio/ML_2023/Data_split/onlytargets_test30.csv', index=False)
