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
    

#Target cols are: X, Y, Z 
Y_train_X= trainDataset['X']
Y_train_Y= trainDataset['Y'] 
Y_train_Z= trainDataset['Z'] 

kf = KFold(n_splits=10, shuffle=True, random_state=42)
#RBF Kernel

# Target X
target = 'X'
alpha_vals = np.logspace(-9, 0, 30, base=2)
gamma_vals = np.logspace(-9, 3, 10, base=2)

param_options = [
    {'alpha': alpha_vals, 'gamma': gamma_vals},
    {'alpha': alpha_vals}
]

# Perform a grid search with cross-validation
krr_search = GridSearchCV(
    KernelRidge(kernel="rbf"),
    param_grid=param_options,
    cv=kf,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Train the model on the provided data
krr_search.fit(X_train, Y_train_X)

# Print the optimal parameters and their associated score
print("For target variable " + target + ", RBF optimal parameters are: %s with a score: %0.5f"% (krr_search.best_params_, krr_search.best_score_))

# Obtain the best estimator with optimal parameters
rbf_krr_X = krr_search.best_estimator_

# Target Y
target = 'Y'
# Train the model on the provided data
krr_search.fit(X_train, Y_train_Y)

# Print the optimal parameters and their associated score
print("For target variable " + target + ", RBF optimal parameters are: %s with a score: %0.5f"% (krr_search.best_params_, krr_search.best_score_))

# Obtain the best estimator with optimal parameters
rbf_krr_Y = krr_search.best_estimator_

# Target Z
target = 'Z'
# Train the model on the provided data
krr_search.fit(X_train, Y_train_Z)

# Print the optimal parameters and their associated score
print("For target variable " + target + ", RBF optimal parameters are: %s with a score: %0.5f"% (krr_search.best_params_, krr_search.best_score_))

# Obtain the best estimator with optimal parameters
rbf_krr_Z = krr_search.best_estimator_

#Linear Kernel

# Target X
target = 'X'

linear_options = [
    {'alpha': alpha_vals}
]

# Perform a grid search with cross-validation
linear_search = GridSearchCV(
    KernelRidge(kernel="linear"),
    param_grid=linear_options,
    cv=kf,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Train the model on the provided data
linear_search.fit(X_train, Y_train_X)

# Print the optimal parameters and their associated score
print("For target variable " + target + ", Linear optimal parameters are: %s with a score: %0.5f"% (linear_search.best_params_, linear_search.best_score_))

# Obtain the best estimator with optimal parameters
linear_krr_X = linear_search.best_estimator_

# Target Y
target = 'Y'
# Train the model on the provided data
linear_search.fit(X_train, Y_train_Y)

# Print the optimal parameters and their associated score
print("For target variable " + target + ", Linear optimal parameters are: %s with a score: %0.5f"% (linear_search.best_params_, linear_search.best_score_))

# Obtain the best estimator with optimal parameters
linear_krr_Y = linear_search.best_estimator_

# Target Z
target = 'Z'
# Train the model on the provided data
linear_search.fit(X_train, Y_train_Z)

# Print the optimal parameters and their associated score
print("For target variable " + target + ", Linear optimal parameters are: %s with a score: %0.5f"% (linear_search.best_params_, linear_search.best_score_))

# Obtain the best estimator with optimal parameters
linear_krr_Z = linear_search.best_estimator_


#Poly Kernel

# Target X
target='X'
degree_range = np.arange(2, 8, 1)
alpha_range = np.logspace(-9, 0, 10, base = 2)
gamma_range = np.logspace(-9, 3, 10, base = 2)

poly_options = [
    {'alpha': alpha_range, 'gamma': gamma_range, 'degree': degree_range},
    {'alpha': alpha_range, 'degree': degree_range}
]

poly_search = GridSearchCV(
    KernelRidge(kernel = 'poly'),
    param_grid = poly_options,
    cv = kf,
    scoring = 'neg_mean_squared_error',
    n_jobs = -1
)

poly_search.fit(X_train, Y_train_X)

# Print the optimal parameters and their associated score
print("For target variable " + target + ", Poly optimal parameters are: %s with a score: %0.5f"% (poly_search.best_params_, poly_search.best_score_))

poly_krr_X = poly_search.best_estimator_

#Target Y
target = 'Y'
poly_search.fit(X_train, Y_train_Y)

# Print the optimal parameters and their associated score
print("For target variable " + target + ", Poly optimal parameters are: %s with a score: %0.5f"% (poly_search.best_params_, poly_search.best_score_))

poly_krr_Y = poly_search.best_estimator_
#Target Z
target = 'Z'
poly_search.fit(X_train, Y_train_Z)

# Print the optimal parameters and their associated score
print("For target variable " + target + ", Poly optimal parameters are: %s with a score: %0.5f"% (poly_search.best_params_, poly_search.best_score_))
poly_krr_Z = poly_search.best_estimator_

#Train on whole dataset

poly_search.fit(X_train, Y_train)
print("For all target variables, Poly optimal parameters are: %s with a score: %0.5f"% (poly_search.best_params_, poly_search.best_score_))
poly_krr= poly_search.best_estimator_
joblib.dump(poly_krr, path.join(path.abspath(path.dirname(__file__)), "results/poly_krr.z"))