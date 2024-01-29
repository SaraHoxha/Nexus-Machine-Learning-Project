#In this code we experiment with the leave one out cross validation technique, the technique did not yield exceptional results, this is because the amount of data that we have.
#Also is important to mention that since LOOCV does validation with only one observation in each fold, the Bias/Variance tradeoff tends to the Variance extreme.
#This can be seen in the results, where we got a variance in accuracy of almost 30%
#Loading libraries
import numpy as np
import pandas as pd
import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import sys
import os.path as path
#Import data
data = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "..", "..", "..", "Data_processing", "normalized_training.csv"))
# Convert data to a NumPy array
data_array = data.to_numpy()
# Get the first ten columns without id
inputs_training = data_array[:, 1:-3]
inputs_training.shape
# Get the last three columns
targets_training = data_array[:, -3:]
targets_training.shape

#####################################
#K-Fold cross validation preparation - with leave one out approach
num_folds = len(inputs_training) -1
# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Define the LOO Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

#Print the output of the algo to a txt
# Redirect output to a file
sys.stdout = open('../ML_2023/NN/Leave-one-out-crossval/terminal_output.txt', 'w')
# LOO Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs_training, targets_training):

    inputs = keras.Input(shape=(10,))
    inputs.shape
    inputs.dtype

    #Simple NN example
    ############################
    #Creating the NN Architecture
    #first hidden layer
    dense = layers.Dense(64, activation="sigmoid")
    #Pass the inputs to the first hidden layer
    x = dense(inputs)
    #Create a second layer
    x = layers.Dense(10, activation="relu")(x)
    #Your output layer
    outputs = layers.Dense(3)(x)
    #############################
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    #model.summary()
    #############################

    #Model training
    ##################
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.SGD(learning_rate=0.0011),
        metrics=["accuracy"],
    )

     # Generate a print with each step of LOO
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    history = model.fit(inputs_training[train], targets_training[train], batch_size=64, epochs=150,workers=-1,verbose=0)

    ###############################################################
    #Final values on test data
    test_scores = model.evaluate(inputs_training[test], targets_training[test], verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {test_scores[0]}; {model.metrics_names[1]} of {test_scores[1]*100}%')
    acc_per_fold.append(test_scores[1] * 100)
    loss_per_fold.append(test_scores[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

# Print some text to the terminal
print('This text will be saved to the file')
# Print a newline character to the terminal
print()
# Redirect output back to the console
sys.stdout = sys.__stdout__