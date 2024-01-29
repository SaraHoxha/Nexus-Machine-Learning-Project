#In this script we run a random forest feature selection model to check the most important attributes for the desired targets.

#NOTES:
    #In this plot we can see as shown in the simple correlation heat map that attribute D is almost not important for the targets.
    #This give us even more reasons to dig deeper with the intention to remove attribute D from the model selection part.

#Load libraries
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

#Import datasets
df_training = pd.read_csv("ML-CUP23-TR.csv")
df_test = pd.read_csv("ML-CUP23-TS.csv")

# Separate features and target variable
X = df_training.drop(columns=["id","X","Y","Z"]) 
y = df_training[["X","Y","Z"]] 

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=400, random_state=42,min_samples_split=3,min_samples_leaf=3,n_jobs=-1,ccp_alpha=0.1)

# Fit the model
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_

# Sort feature importances in descending order
indices = importances.argsort()[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()
