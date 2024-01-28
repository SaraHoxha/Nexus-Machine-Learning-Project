import tensorflow as tf
import os.path as path
import pandas as pd
import joblib

def mean_euclidian_error(y_true, y_pred):
    errors = tf.sqrt(tf.reduce_sum((y_pred - y_true)**2, axis=-1))
    mean_euclidean_error = tf.reduce_mean(errors)
    return mean_euclidean_error

def getTestDataSetPath ():
    return path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "Data_processing", "normalized_test.csv")

testDataset = pd.read_csv(getTestDataSetPath())

#KNR Test
model_path_knr= path.join(path.abspath(path.dirname(__file__)), "..","knr/results/knrBestModel.z")
model_knr= joblib.load(model_path_knr)
X_test = testDataset.iloc[:, 1:11]
Y_predicted_knr = model_knr.predict(X_test)
labels_path_knr = path.join(path.dirname(__file__), "results", "knrTestLabels.csv")
pd.DataFrame(Y_predicted_knr, columns=["X", "Y", "Z"]).to_csv(labels_path_knr, index=True)

#KRR Test
model_path_krr = path.join(path.abspath(path.dirname(__file__)), "..","krr/results/poly_krr.z")
model_krr = joblib.load(model_path_krr)
X_test = testDataset.iloc[:, 1:11]
Y_predicted_krr = model_krr.predict(X_test)
labels_path_krr = path.join(path.dirname(__file__), "results", "krrTestLabels.csv")
pd.DataFrame(Y_predicted_krr, columns=["X", "Y", "Z"]).to_csv(labels_path_krr, index=True)

#Random Forest Test
model_path_rf = path.join(path.abspath(path.dirname(__file__)), "..","random forest/results/randomForestBestModel.z")
model_rf = joblib.load(model_path_rf)
X_test = testDataset.iloc[:, 1:11]
Y_predicted_fr = model_rf.predict(X_test)
labels_path_fr = path.join(path.dirname(__file__), "results", "randomForestTestLabels.csv")
pd.DataFrame(Y_predicted_fr, columns=["X", "Y", "Z"]).to_csv(labels_path_fr, index=True)

#Linear Test
model_path_linear = path.join(path.abspath(path.dirname(__file__)), "..","linear/results/linearBestModel.z")
model_linear = joblib.load(model_path_linear)
X_test = testDataset.iloc[:, 1:11]
Y_predicted_linear = model_linear.predict(X_test)
labels_path_linear = path.join(path.dirname(__file__), "results", "linearTestLabels.csv")
pd.DataFrame(Y_predicted_linear, columns=["X", "Y", "Z"]).to_csv(labels_path_linear, index=True)

#Lasso Test
model_path_lasso = path.join(path.abspath(path.dirname(__file__)), "..","linear/results/lassoBestModel.z")
model_lasso = joblib.load(model_path_lasso)
X_test = testDataset.iloc[:, 1:11]
Y_predicted_lasso = model_lasso.predict(X_test)
labels_path_lasso = path.join(path.dirname(__file__), "results", "lassoTestLabels.csv")
pd.DataFrame(Y_predicted_lasso, columns=["X", "Y", "Z"]).to_csv(labels_path_lasso, index=True)

#Ridge Test
model_path_ridge = path.join(path.abspath(path.dirname(__file__)), "..","linear/results/ridgeBestModel.z")
model_ridge = joblib.load(model_path_ridge)
X_test = testDataset.iloc[:, 1:11]
Y_predicted_ridge = model_ridge.predict(X_test)
labels_path_ridge = path.join(path.dirname(__file__), "results", "ridgeTestLabels.csv")
pd.DataFrame(Y_predicted_ridge, columns=["X", "Y", "Z"]).to_csv(labels_path_ridge, index=True)

