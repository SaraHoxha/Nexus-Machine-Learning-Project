import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from krrUtils import getTrainDatasetPath
import joblib
import os.path as path

trainDataset = pd.read_csv(getTrainDatasetPath())

X_train = trainDataset.iloc[:, 1:11]
Y_train = trainDataset.iloc[:, 11:14]

kf = KFold(n_splits=10, shuffle=True, random_state=42)

alpha_vals = np.logspace(-9, 0, 30, base=2)
gamma_vals = np.logspace(-9, 3, 10, base=2)


degree_range = np.arange(2, 8, 1)
alpha_range = np.logspace(-9, 0, 10, base = 2)
gamma_range = np.logspace(-9, 3, 10, base = 2)

param_options = {
    'alpha': alpha_range,  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel types
    'gamma': gamma_range,  # Gamma for RBF kernel
    'degree': degree_range, # Degree for polynomial kernel
}

# Perform a grid search with cross-validation
krr_search = GridSearchCV(
    KernelRidge(kernel="rbf"),
    param_grid=param_options,
    cv=kf,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Train the model on the provided data
krr_search.fit(X_train, Y_train)

# Print the optimal parameters and their associated score
print("The optimal parameters are: %s with a score: %0.5f"% (krr_search.best_params_, krr_search.best_score_))

krr= krr_search.best_estimator_
joblib.dump(krr, path.join(path.abspath(path.dirname(__file__)), "results/KrrBestModel.z"))