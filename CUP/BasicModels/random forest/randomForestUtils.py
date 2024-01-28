import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.utils import resample

def learningCurveForDifferentDatasetSize(dataset_sizes, mee_values, validationAcc):

    # Plotting the learning curves for Mean Euclidian Error
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, mee_values, marker='o', label='Mean Euclidian Error', color='red')
    plt.xlabel('Dataset Size')
    plt.ylabel('Mean Euclidian Error')
    plt.title('Random Forest Learning Curve for Mean Euclidian Error')
    plt.legend()
    plt.grid(True)
    plt.xticks(dataset_sizes)  # Set x-axis ticks to dataset sizes
    plt.savefig(path.join(path.dirname(__file__), "plots", "learningCurve_MEE.png"))

    # Plotting the learning curves for validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, validationAcc, marker='o', label='Validation Accuracy', color='blue')
    plt.xlabel('Dataset Size')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Learning Curve for Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(dataset_sizes)  # Set x-axis ticks to dataset sizes
    plt.savefig(path.join(path.dirname(__file__), "plots", "learningCurve_ValidationAccuracy.png"))
    
    # Function to downsample a dataset
def downsample_and_get_datasets(dataset: pd.DataFrame, random_state=69) -> dict:
    def downsample_dataset(dataset, n):
        resampled_array = resample(dataset, n_samples=n, replace=False, random_state=random_state)
        return pd.DataFrame(resampled_array, columns=dataset.columns)

    def get_sample_size_list(number):
        percentages = [math.ceil(number * i / 100) for i in range(10, 101, 10)]
        return percentages

    downsampled_datasets = {}
    for dataset_dimension in get_sample_size_list(dataset.shape[0]):
        downsampled_datasets[dataset_dimension] = downsample_dataset(dataset, dataset_dimension)

    return downsampled_datasets

def getTrainDatasetPath ():
    return path.join(path.abspath(path.dirname(__file__)), "..", "..",  "..","Data_processing", "train70.csv")

def getTestDatasetPath ():
    return path.join(path.abspath(path.dirname(__file__)), "..", "..",  "..","Data_processing", "test30.csv")

def plot3D(predictions, ground_truth, phase):
 
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for predicted values 
    ax.scatter(predictions["X"], predictions["Y"], predictions["Z"], c='r', marker='o', label='Predicted')

    # Scatter plot for validation values 
    ax.scatter(ground_truth["X"], ground_truth["Y"], ground_truth["Z"], c='b', marker='o', label='True')

    # Set labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Set title
    ax.set_title('Random Forest 3D Scatter Plot ' + str(phase))

    # Add legend
    ax.legend()

    plt.savefig(path.join(path.dirname(__file__), "plots", "randomForest3DPlot.png"))