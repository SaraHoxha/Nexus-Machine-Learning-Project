#In this file we explore and evaluate the best models obtained in the 'complex' and 'simpler' searches
#We produce the predicted labels for the training and for the test
#We also produced the learning curves for each of the models
#Important note, here we don't split the training data anymore since we are not going to change anything from the model.

#Load libraries
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import the datasets
data = pd.read_csv("Data_processing/train70.csv")
data_test = pd.read_csv("Data_processing/test30.csv")
# Convert data to a NumPy array
data_array = data.to_numpy()
data_test_array = data_test.to_numpy()
#Data process
#############################
# Get the first ten columns without id
inputs_training = data_array[:, 1:-3]
inputs_training.shape
# Get the first ten columns without id
inputs_test = data_test_array[:, 1:-3]
inputs_test.shape
# Get the last three columns
targets_training = data_array[:, -3:]
targets_training.shape
# Get the last three columns
targets_test = data_test_array[:, -3:]
targets_test.shape

#Number of best models that will be loaded for evaluation
num_models = 10
models = []
for i in range(num_models):
#Reconstructing each of the best methods
    reconstructed_model = keras.models.load_model(f'CUP/NN/NNselection/complex/Models/keras/model_{i}.keras')
    models.append(reconstructed_model)
#Re train, extract the predicted values, plot the final learning curves of each of the 10 best models
for i in range(num_models):
    print (f'Model_{i}')
    model = models[i]
    history = model.fit(inputs_training, targets_training, batch_size=64, epochs=70, validation_split=0.2, validation_data=(inputs_test,targets_test),workers=-1,verbose=1,shuffle=True)
    #Save the predicted labels for future plotting
    training_pred_labels = model.predict(inputs_training, batch_size=None, verbose="auto", steps=None, callbacks=None)
    # Save the array as a CSV file
    np.savetxt(f"CUP/NN/NNselection/complex/Predicted_vals_best_models/training/training_pred_labels_model_{i}.csv", training_pred_labels, delimiter=",")
    test_pred_labels = model.predict(inputs_test, batch_size=None, verbose="auto", steps=None, callbacks=None)
    # Save the array as a CSV file
    np.savetxt(f"CUP/NN/NNselection/complex/Predicted_vals_best_models/test/test_pred_labels_model_{i}.csv", test_pred_labels, delimiter=",")
    #Plotting the results
    ##############################################
    # Create a plot of the training loss and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss(MSE)')
    plt.legend()
    plt.title(f'Training curves for loss complex models CUP model {i}')
    plt.savefig(f'CUP/NN/NNselection/complex/Plots/Loss/complex_loss_model_{i}.png')
    #plt.show()
    plt.clf()

    # Create a plot of the training accuracy and validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Training curves for accuracy complex models CUP model {i}')
    plt.savefig(f'CUP/NN/NNselection/complex/Plots/Acc/complex_acc_model_{i}.png')
    #plt.show()
    plt.clf()