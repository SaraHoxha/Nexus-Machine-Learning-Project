#In this file we created the initial split to have a real test subset, with the train70 we will do training and validation for the models
#With the test30 we will do the final internal test when the models are ready and there is nothing else to change in the models.

#Import libraries
import numpy as np
import pandas as pd
import os.path as path

#Import the normalized data set(Only inputs are normalized, the outputs are not)
df = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "Data_processing", "normalized_training.csv"))

# Set the random seed
seed = 42
np.random.seed(seed)

# Shuffle the DataFrame's index
df_index = df.index.to_numpy()
np.random.shuffle(df_index)

# Split the DataFrame into two based on the shuffled index
split_index = int(0.7 * len(df_index))
train_index = df_index[:split_index]
test_index = df_index[split_index:]

# Create the training and test sets
train_df = df.loc[train_index]
test_df = df.loc[test_index]

#Print and check the splits
test_df
train_df

#Save the splits
train_df.to_csv(path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "Data_processing", "train70.csv"), index=False)
test_df.to_csv(path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "Data_processing", "test30.csv"), index=False)
