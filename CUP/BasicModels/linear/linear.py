import joblib
import pandas as pd
import os.path as path
from sklearn.linear_model import LinearRegression
from linearUtils import getTrainDatasetPath

# Read the training dataset
trainDataset = pd.read_csv(getTrainDatasetPath())
X_train = trainDataset.iloc[:, 1:11]
Y_train = trainDataset.iloc[:, 11:14]

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
linearBestModel= model.fit(X_train, Y_train)

file_path = path.join(path.abspath(path.dirname(__file__)), "results/linearBestModel.z")
joblib.dump(linearBestModel, file_path)