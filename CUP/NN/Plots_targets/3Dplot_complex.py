#In this file we create the plots to check the predicted labels against the ground truth, this is to check if we are actually fitting the behavior of the targets or just noise points.

#Loading libraries
import pandas as pd
import sys
sys.path.append('ML_PROJECT_FINAL')
from plot_valid_pred import Plot_Valid_Pred

#Importing data
true_targets_training = pd.read_csv("Data_processing/onlytargets_train70.csv")
true_targets_test = pd.read_csv("Data_processing/onlytargets_test30.csv")

#Path and number of models to save
path_to_save = "CUP/NN/Plots_targets/complex"
num_models = 10

for i in range(num_models):
    pred_targets_training = pd.read_csv(f"CUP/NN/NNselection/complex/Predicted_vals_best_models/training/training_pred_labels_model_{i}.csv")
    pred_targets_test = pd.read_csv(f"CUP/NN/NNselection/complex/Predicted_vals_best_models/test/test_pred_labels_model_{i}.csv")
    # Add titles to each column
    pred_targets_training.columns = ["X", "Y", "Z"]
    pred_targets_test.columns = ["X", "Y", "Z"]
    Plot_Valid_Pred(pred_targets_training,true_targets_training,path_to_save,f"training_model_{i}")
    Plot_Valid_Pred(pred_targets_test,true_targets_test,path_to_save,f"test_model_{i}")