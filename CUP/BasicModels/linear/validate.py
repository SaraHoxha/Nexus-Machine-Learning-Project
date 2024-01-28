import tensorflow as tf
import os.path as path
import pandas as pd
import joblib
from linearUtils import getTestDatasetPath, getTrainDatasetPath, downsample_and_get_datasets, plotLearningCurves, plot3D

def mean_euclidian_error(y_true, y_pred):
    errors = tf.sqrt(tf.reduce_sum((y_pred - y_true)**2, axis=-1))
    mean_euclidean_error = tf.reduce_mean(errors)
    return mean_euclidean_error

trainDataset = pd.read_csv(getTrainDatasetPath())
internalTestDataset = pd.read_csv(getTestDatasetPath())

linear_file_path = path.join(path.abspath(path.dirname(__file__)), "results/linearBestModel.z")
lasso_file_path = path.join(path.abspath(path.dirname(__file__)), "results/lassoBestModel.z")
ridge_file_path = path.join(path.abspath(path.dirname(__file__)), "results/ridgeBestModel.z")

linearModel = joblib.load(linear_file_path)
lassoModel = joblib.load(lasso_file_path)
ridgeModel = joblib.load(ridge_file_path)

downsampledDatasets = downsample_and_get_datasets(internalTestDataset)


#Predicting on whole internal test
X = internalTestDataset.iloc[:, 1:11]
Y = internalTestDataset.iloc[:, 11:14]


Y_predicted_linear = linearModel.predict(X)
Y_predicted_lasso = lassoModel.predict(X)
Y_predicted_ridge = ridgeModel.predict(X)



Y_predicted_linear_df = pd.DataFrame(linearModel.predict(X), columns=["X", "Y", "Z"])
Y_predicted_lasso_df = pd.DataFrame(lassoModel.predict(X), columns=["X", "Y", "Z"])
Y_predicted_ridge_df = pd.DataFrame(ridgeModel.predict(X), columns=["X", "Y", "Z"])

print("Linear Accuracy is " + str(linearModel.score(X, Y_predicted_linear))) 
print("Linear MEE is " + str(mean_euclidian_error(Y, Y_predicted_linear))) 

print("Lasso Accuracy is " + str(lassoModel.score(X, Y_predicted_lasso))) 
print("Lasso MEE is " + str(mean_euclidian_error(Y, Y_predicted_lasso))) 

print("Ridge Accuracy is " + str(lassoModel.score(X, Y_predicted_ridge)))
print("Ridge MEE is " + str(mean_euclidian_error(Y, Y_predicted_ridge)))

mee_scores_list, validation_accuracy_list, dataset_sizes = [],[], []
mee_scores_list_lasso, validation_accuracy_list_lasso = [],[]
mee_scores_list_ridge, validation_accuracy_list_ridge = [],[]

#Predicting on different internal test dataset sizes
for datasetDimension, subDataset in downsampledDatasets.items():
        X_test = subDataset.iloc[:, 1:11]
        Y_test = subDataset.iloc[:, 11:14]
        
        predicted_Y_linear = linearModel.predict(X_test)
        predicted_Y_lasso = lassoModel.predict(X_test)
        predicted_Y_ridge = ridgeModel.predict(X_test)
        
        mee_scores_list.append(mean_euclidian_error(Y_test, predicted_Y_linear))
        mee_scores_list_lasso.append(mean_euclidian_error(Y_test, predicted_Y_lasso))
        mee_scores_list_ridge.append(mean_euclidian_error(Y_test, predicted_Y_ridge))
        
        validation_accuracy_list.append(linearModel.score(X_test, Y_test))
        validation_accuracy_list_lasso.append(lassoModel.score(X_test, Y_test))
        validation_accuracy_list_ridge.append(ridgeModel.score(X_test, Y_test))
        
        dataset_sizes.append(datasetDimension)

plotLearningCurves("Linear", dataset_sizes, mee_scores_list, validation_accuracy_list)
plotLearningCurves("Lasso", dataset_sizes, mee_scores_list_lasso, validation_accuracy_list_lasso)
plotLearningCurves("Ridge", dataset_sizes, mee_scores_list_ridge, validation_accuracy_list_ridge)

plot3D(Y_predicted_linear_df, Y, "Validation", "Linear")
plot3D(Y_predicted_ridge_df, Y, "Validation", "Ridge")
plot3D(Y_predicted_lasso_df, Y, "Validation", "Lasso")