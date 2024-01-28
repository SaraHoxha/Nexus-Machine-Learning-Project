#In this file we predict the blind test provided by the professor and export the predicted targets for the ML CUP

#Load libraries
import keras
import pandas as pd
import numpy as np
import sys

#Import data
data = pd.read_csv("C:/Users/urbi1/OneDrive/Escritorio/ML_2023/NN/normalized_training.csv")
data_test = pd.read_csv("C:/Users/urbi1/OneDrive/Escritorio/ML_2023/NN/normalized_test.csv")
# Convert data to a NumPy array
data_array = data.to_numpy()
data_test_array = data_test.to_numpy()
#Data process
#############################
# Get the first ten columns without id
inputs_training = data_array[:, 1:-3]
inputs_training.shape
# Get the first ten columns without id
inputs_test = data_test_array[:, 1:]
inputs_test.shape
# Get the last three columns
targets_training = data_array[:, -3:]
targets_training.shape

# Capture the terminal output
old_stdout = sys.stdout

# Create a file object to write to
file_object = open("NN/FinalNNselection/Results/CUP/complex/cup_terminal.txt", "w")

# Set sys.stdout to the file object
sys.stdout = file_object

#Number of best models that will be loaded for test
num_models = 10
models = []
for i in range(num_models):
#Reconstructing each of the best methods
    reconstructed_model = keras.models.load_model(f'NN/FinalNNselection/Results/Models/keras/model_{i}.keras')
    models.append(reconstructed_model)
#Test the predictions on the blind test for the ML CUP
for i in range(num_models):
    print (f'Model_{i}')
    model = models[i]
    history = model.fit(inputs_training, targets_training, batch_size=64, epochs=70, validation_split=0.05,workers=-1,verbose=1,shuffle=True)
    #Save the predicted labels for future plotting
    test_pred_labels = model.predict(inputs_test, batch_size=None, verbose="auto", steps=None, callbacks=None)
    # Save the array as a CSV file
    np.savetxt(f"NN/FinalNNselection/Results/CUP/complex/CUP_pred_labels_model_{i}.csv", test_pred_labels, delimiter=",",header="X,Y,Z")

# Restore sys.stdout
sys.stdout = old_stdout

# Close the file object
file_object.close()