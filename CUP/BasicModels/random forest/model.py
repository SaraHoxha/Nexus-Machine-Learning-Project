import pandas as pd
import numpy as np
import joblib
import os.path as path
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from randomForestUtils import  getTrainDatasetPath
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def calculateMetrics(predictions, groundTruth):
    # R2 Score
    r2 = r2_score(groundTruth, predictions)

    # Mean Squared Error (MSE)
    mse = mean_squared_error(groundTruth, predictions)

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(groundTruth, predictions)

    return {
        "r2Score": r2,
        "meanSquaredError": mse,
        "meanAbsoluteError": mae,
    }

trainDataset = pd.read_csv(getTrainDatasetPath())
X = trainDataset.iloc[:, 1:11]
Y = trainDataset.iloc[:, 11:14]

kf = KFold(n_splits=10, shuffle=True, random_state=42)

best_model_info = {
        'model': None,
        'metrics': None,
        'n_estimators': None,
        'max_depth': None,
    }

param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20, 30]
            }

metrics_list = [] # List to store metrics for each fold

for trainIndex, testIndex in kf.split(X):
    X_train, X_test = X.iloc[trainIndex], X.iloc[testIndex]
    Y_train, Y_test = Y.iloc[trainIndex], Y.iloc[testIndex]
        
    grid_search = GridSearchCV(RandomForestRegressor(n_jobs=-1), param_grid, cv=kf, scoring='neg_mean_squared_error', return_train_score=False)
    grid_search.fit(X_train, Y_train)

    # Extracting the best model from the grid search
    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)
    metrics = calculateMetrics(predictions=predictions, groundTruth=Y_test)
    metrics_list.append(metrics)

avg_metrics = {
                "r2Score": np.mean([m["r2Score"] for m in metrics_list]),
                "meanSquaredError": np.mean([m["meanSquaredError"] for m in metrics_list]),
                "meanAbsoluteError": np.mean([m["meanAbsoluteError"] for m in metrics_list]),
    }

if best_model_info['model'] is None or avg_metrics['meanSquaredError'] < best_model_info['metrics']['meanSquaredError']:
    best_model_info['model'] = best_model
    best_model_info['metrics'] = avg_metrics
    best_model_info['max_depth'] = best_model.max_depth
    best_model_info['n_estimators'] = best_model.n_estimators


print(f"Best Model Information:\n"
      f"  - Estimators: {best_model_info['n_estimators']}\n"
      f"  - Max Depth: {best_model_info['max_depth']}\n"
      f"  - Metrics: {best_model_info['metrics']}")

#Retrain with full training set
randomForestBestModel = RandomForestRegressor(n_estimators=best_model_info['n_estimators'], max_depth=best_model_info['max_depth']).fit(X,Y)

file_path = path.join(path.abspath(path.dirname(__file__)), "results/randomForestBestModel.z")
joblib.dump(randomForestBestModel, file_path)