import tensorflow as tf
import os.path as path
import pandas as pd
import joblib
from krrUtils import getTestDatasetPath, getTrainDatasetPath, downsample_and_get_datasets, learningCurveForDifferentDatasetSize, plot3D

def mean_euclidian_error(y_true, y_pred):
    errors = tf.sqrt(tf.reduce_sum((y_pred - y_true)**2, axis=-1))
    mean_euclidean_error = tf.reduce_mean(errors)
    return mean_euclidean_error

trainDataset = pd.read_csv(getTrainDatasetPath())
internalTestDataset = pd.read_csv(getTestDatasetPath())

file_path = path.join(path.abspath(path.dirname(__file__)), "results/poly_krr.z")
model = joblib.load(file_path)

downsampledDatasets = downsample_and_get_datasets(internalTestDataset)

mee_scores_list, validation_accuracy_list, dataset_sizes = [],[], []

#Predicting on whole internal test
X = internalTestDataset.iloc[:, 1:11]
Y = internalTestDataset.iloc[:, 11:14]
Y_predicted = model.predict(X)
Y_predicted_df = pd.DataFrame(model.predict(X), columns=["X", "Y", "Z"])

print("Accuracy is " + str(model.score(X, Y)))
print("MEE is " + str(mean_euclidian_error(Y, Y_predicted)))

#Predicting on different internal test dataset sizes
for datasetDimension, subDataset in downsampledDatasets.items():
        X_test = subDataset.iloc[:, 1:11]
        Y_test = subDataset.iloc[:, 11:14]
        
        predicted_Y = model.predict(X_test)
        
        mee_scores_list.append(mean_euclidian_error(Y_test, predicted_Y))
        validation_accuracy_list.append(model.score(X_test, Y_test))
        dataset_sizes.append(datasetDimension)


learningCurveForDifferentDatasetSize(dataset_sizes, mee_scores_list, validation_accuracy_list)
plot3D(Y_predicted_df , Y, "Validation")
