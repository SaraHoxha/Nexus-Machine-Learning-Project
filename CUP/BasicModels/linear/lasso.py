import pandas as pd
import numpy as np
import joblib
import os.path as path
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, GridSearchCV
from linearUtils import  getTrainDatasetPath
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
        'alpha': None,
    }

param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}

metrics_list = [] # List to store metrics for each fold

for trainIndex, testIndex in kf.split(X):
    X_train, X_test = X.iloc[trainIndex], X.iloc[testIndex]
    Y_train, Y_test = Y.iloc[trainIndex], Y.iloc[testIndex]
        
    grid_search = GridSearchCV(Lasso(), param_grid, cv=kf, scoring='neg_mean_squared_error', return_train_score=False)
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
    best_model_info['alpha'] = best_model.alpha


print(f"Best Model Information:\n"
      f"  - Alpha: {best_model_info['alpha']}\n"
      f"  - Metrics: {best_model_info['metrics']}")

#Retrain with full training set
lassoBestModel = Lasso(alpha=best_model_info['alpha']).fit(X,Y)

file_path = path.join(path.abspath(path.dirname(__file__)), "results/lassoBestModel.z")
joblib.dump(lassoBestModel, file_path)
