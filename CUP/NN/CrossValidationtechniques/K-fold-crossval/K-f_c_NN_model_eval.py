#In this file we explore the K fold cross validation technique in order to use it for the project, the results were satisfactory for the CUP models, but not so much for the Monks problems
#This can happen because of the amount of observations, in Monks the observations were on the few hundreds, and so, the model was not containing sufficient amount of observation in each validation fold.
#For the CUP in the other hand the results for were much better and was one of the measures used to check if the best models selected by the tuner really yielded models that performed well, this K fold technique yielded results for our models ranging from 92% to 98% in the best NN models selected these results also had a very low variance of 1% or 2%

#Loading libraries
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import KFold

#Import data
data = pd.read_csv("C:/Users/urbi1/OneDrive/Escritorio/ML_2023/NN/normalized_training.csv")
# Convert data to a NumPy array
data_array = data.to_numpy()
# Get the first ten columns without id
inputs_training = data_array[:, 1:-3]
inputs_training.shape
# Get the last three columns
targets_training = data_array[:, -3:]
targets_training.shape

#####################################
#K-Fold cross validation preparation
num_folds = 10
# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs_training, targets_training):

    inputs = keras.Input(shape=(10,))
    #Basic experimental architecture
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
    model.summary()
    #############################

    #Model training
    ##################
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.SGD(learning_rate=0.0011),
        metrics=["accuracy"],
    )

     # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    history = model.fit(inputs_training[train], targets_training[train], batch_size=64, epochs=350,workers=-1)

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