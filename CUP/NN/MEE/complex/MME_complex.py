import pandas as pd
import numpy as np
import csv
import sys
sys.path.append('ML_PROJECT_FINAL/CustomLoss')
from MEE import MEE

MEE_losses = {}
num_models = 10
for i in range(num_models):
    pred_test = pd.read_csv(f'CUP/NN/NNselection/complex/Predicted_vals_best_models/test_pred_labels_model_{i}.csv',names=['X', 'Y', 'Z'])
    # Rename the column names
    true_test = pd.read_csv('Data_split/onlytargets_test30.csv')
    mee_loss = MEE(true_test,pred_test)
    MEE_losses[f'model_{i}'] = mee_loss

with open('CUP/NN/MEE/complex/test_models.csv', 'w', newline='') as f:
    # Create a CSV writer object
    writer = csv.DictWriter(f, fieldnames=MEE_losses.keys())

    # Write the header row
    writer.writeheader()

    # Write the data row
    writer.writerow(MEE_losses)
