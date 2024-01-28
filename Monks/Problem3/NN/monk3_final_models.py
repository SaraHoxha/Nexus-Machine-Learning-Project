#In this file me use the 5 best selected models to solve the Monk problem at hand, we run the model, save the predictions and save the learning plots.
#Load libraries
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Import datasets
data = pd.read_csv("C:/Users/urbi1/OneDrive/Escritorio/ML_2023/monk+s+problems/encoded_monks-3train.csv")
data_test = pd.read_csv("C:/Users/urbi1/OneDrive/Escritorio/ML_2023/monk+s+problems/encoded_monks-3test.csv")
# Convert data to a NumPy array
data_array = data.to_numpy()
data_test_array = data_test.to_numpy()
#Data process
#############################
#Get the 17 inputs
inputs_training = data_array[:, 3:]
inputs_training.shape
inputs_training = np.asarray(inputs_training).astype('int')
#Get the binary class
targets_training = data_array[:,1]
targets_training.shape
targets_training = np.asarray(targets_training).astype('int')
#Get the 17 inputs
inputs_test = data_test_array[:, 3:]
inputs_test.shape
inputs_test = np.asarray(inputs_test).astype('int')
#Get the binary class
targets_test = data_test_array[:,1]
targets_test.shape
targets_test = np.asarray(targets_test).astype('int')

#Number of best models that will be loaded for evaluation
num_models = 5
models = []
for i in range(num_models):
#Reconstructing each of the best methods
    reconstructed_model = keras.models.load_model(f'monk+s+problems/Problem3/Models/keras/model_{i}.keras')
    models.append(reconstructed_model)
#Re train, extract the predicted values, plot the final learning curves of each of the n best models
for i in range(num_models):
    print (f'Model_{i}')
    model = models[i]
    history = model.fit(inputs_training, targets_training, epochs=100, validation_split=0.2, validation_data=(inputs_test,targets_test),workers=-1,verbose=1,shuffle=True)
    #Save the predicted labels for future plotting
    training_pred_labels = model.predict(inputs_training, batch_size=None, verbose="auto", steps=None, callbacks=None)
    # Save the array as a CSV file
    np.savetxt(f"monk+s+problems/Problem3/predictedlabels/training_pred_labels_model_{i}.csv", training_pred_labels, delimiter=",")
    test_pred_labels = model.predict(inputs_test, batch_size=None, verbose="auto", steps=None, callbacks=None)
    # Save the array as a CSV file
    np.savetxt(f"monk+s+problems/Problem3/predictedlabels/test_pred_labels_model_{i}.csv", test_pred_labels, delimiter=",")
    #Plotting the results
    ##############################################
    # Create a plot of the training loss and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss(X binary entropy)')
    plt.legend()
    plt.title('Training curves for loss Monk3')
    plt.savefig(f'monk+s+problems/Problem3/plots/loss_model_{i}.png')
    #plt.show()
    plt.clf()

    # Create a plot of the training accuracy and validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training curves for accuracy Monk3')
    plt.savefig(f'monk+s+problems/Problem3/plots/acc_model_{i}.png')
    #plt.show()
    plt.clf()