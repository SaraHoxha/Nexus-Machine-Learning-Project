#In this file we explore the importance of the independent attributes for the targets, this was made through a Kbest model
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd

#Import the data
data = pd.read_csv("mainDatasets/ML-CUP23-TR.csv")
#
data_array = data.to_numpy()
#Data process
#############################
# Get the first ten columns without id
X = data_array[:, 1:-3]
# Get the last three columns
y = data_array[:, -3:]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SelectKBest with f_regression scoring function (for regression)
k_best = SelectKBest(score_func=f_regression, k=5)  # Choose the number of top features (k)

# Fit SelectKBest to the training data and transform the features
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

# Get the indices of the selected features
selected_indices = k_best.get_support(indices=True)

# Print the indices of the selected features
print("Indices of selected features:", selected_indices)

# Initialize the RandomForestRegressor using the selected features
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model using the selected features
rf.fit(X_train_kbest, y_train)

# Evaluate the model using the selected features
train_score = rf.score(X_train_kbest, y_train)
test_score = rf.score(X_test_kbest, y_test)

print(f"Train R^2 Score with selected features: {train_score:.4f}")
print(f"Test R^2 Score with selected features: {test_score:.4f}")
