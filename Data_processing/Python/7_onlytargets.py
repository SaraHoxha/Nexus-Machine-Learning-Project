#In this file we exclude the targets to then test against the predicted targets
#This is done in the end of the project when the final models are ready

#Load libraries
import pandas as pd
import os.path as path

#Import the data sets
data = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "Data_processing", "train70.csv"))
data_test = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "Data_processing", "test30.csv"))
#Slice the data frames to only leave the targets
targets_training = data.iloc[:, -3:]
targets_test = data_test.iloc[:, -3:]

#Print and check the targets
targets_training
targets_test

#Save the targets
targets_training.to_csv(path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "Data_processing", "onlytargets_train70.csv"), index=False)
targets_test.to_csv(path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "Data_processing", "onlytargets_test30.csv"), index=False)
