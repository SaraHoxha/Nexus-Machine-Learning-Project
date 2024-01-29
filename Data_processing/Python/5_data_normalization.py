import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read the CSV file into a Pandas DataFrame
df_training = pd.read_csv("mainDatasets/ML-CUP23-TR.csv")

df_test = pd.read_csv("mainDatasets/ML-CUP23-TS.csv")

scaler = MinMaxScaler()

transformed_data_training = scaler.fit_transform(df_training.iloc[:, 1:-3])
transformed_data_test = scaler.fit_transform(df_test.iloc[:, 1:])
transformed_data_training
df_training.iloc[:, 1:-3] = transformed_data_training
df_test.iloc[:, 1:] = transformed_data_test

df_training.to_csv('Data_processingnormalized_training.csv', index=False)
df_test.to_csv('Data_processingnormalized_test.csv', index=False)

