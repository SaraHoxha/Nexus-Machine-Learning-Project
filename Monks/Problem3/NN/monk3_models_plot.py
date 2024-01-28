#Important note, here we don't split the training data anymore since we are not going to change anything from the model.
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/urbi1/OneDrive/Escritorio/ML_2023/monk+s+problems/encoded_monks-3train.csv")
data_test = pd.read_csv("C:/Users/urbi1/OneDrive/Escritorio/ML_2023/monk+s+problems/encoded_monks-3test.csv")
# Convert data to a NumPy array
data_array = data.to_numpy()
data_test_array = data_test.to_numpy()
#Data process
#############################
# Get the first ten columns without id
inputs_training = data_array[:, 3:]
inputs_training.shape
inputs_training = np.asarray(inputs_training).astype('int')
# Get the last three columns
targets_training = data_array[:,1]
targets_training.shape
targets_training = np.asarray(targets_training).astype('int')
# Get the first ten columns without id
inputs_test = data_test_array[:, 3:]
inputs_test.shape
inputs_test = np.asarray(inputs_test).astype('int')
# Get the last three columns
targets_test = data_test_array[:,1]
targets_test.shape
targets_test = np.asarray(targets_test).astype('int')

#Number of best models that will be loaded for evaluation
num_models = 5
models = []
losses_list = []
val_losses_list = []
accuracies_list = []
val_accuracies_list = []
for i in range(num_models):
#Reconstructing each of the best methods
    reconstructed_model = keras.models.load_model(f'monk+s+problems/Problem3/Models/keras/model_{i}.keras')
    models.append(reconstructed_model)
#Re train, extract the predicted values and losses
for i in range(num_models):
    print (f'Model_{i}')
    model = models[i]
    history = model.fit(inputs_training, targets_training, batch_size=64, epochs=250, validation_split=0.2, validation_data=(inputs_test,targets_test),workers=-1,verbose=1,shuffle=True)
    ##############################################
    #Save the losses and accuracies to plot them together
    losses_list.append(history.history['loss'])
    val_losses_list.append(history.history['val_loss'])
    accuracies_list.append(history.history['accuracy'])
    val_accuracies_list.append(history.history['val_accuracy'])

model_names = ["MLP0","MLP1","MLP2","MLP3","MLP4"]


#Plot for the losses
# Create a line plot for each model
plt.figure(figsize=(10, 6))
for i, name in enumerate(model_names):
    plt.plot(losses_list[i], label=name, marker='x')
    plt.plot(val_losses_list[i], label=name, marker='v',linestyle='--')

# Customize the plot
plt.title("Learning Curves of 5 best monk3 NN Models(Loss)")
plt.xlabel("Epochs")
plt.ylabel("Loss(X binary entropy)")
plt.legend(loc='upper right')
plt.grid(True)
legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
plt.legend(legend_handles, legend_labels, title="Model Type")
#plt.show()
plt.savefig('monk+s+problems/Problem3/slidesplots/loss_monk3_models5.png')
plt.clf

#################################################################

#Plot for the accuracies
# Create a line plot for each model
plt.figure(figsize=(10, 6))
for i, name in enumerate(model_names):
    plt.plot(accuracies_list[i], label=name, marker='x')
    plt.plot(val_accuracies_list[i], label=name, marker='v',linestyle='--')

# Customize the plot
plt.title("Learning Curves of 5 best monk3 NN Models(Accuracy)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='upper right')
plt.grid(True)
#plt.show()
plt.savefig('monk+s+problems/Problem3/slidesplots/acc_monk3_models5.png')
plt.clf