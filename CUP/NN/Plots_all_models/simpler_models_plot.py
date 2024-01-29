#In this file we plot all the models losses and accuracies together, this is in order to check the behavior of the different models and their performance

#Load libraries
import keras
import pandas as pd
import matplotlib.pyplot as plt

#Import data
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
num_models = 20
models = []
losses_list = []
val_losses_list = []
accuracies_list = []
val_accuracies_list = []
for i in range(num_models):
#Reconstructing each of the best methods
    reconstructed_model = keras.models.load_model(f'CUP/NN/NNselection/simpler/Models/keras/model_{i}.keras')
    models.append(reconstructed_model)
#Re train, extract the predicted values and losses
for i in range(num_models):
    print (f'Model_{i}')
    model = models[i]
    history = model.fit(inputs_training, targets_training, batch_size=64, epochs=70, validation_split=0.2, validation_data=(inputs_test,targets_test),workers=-1,verbose=1,shuffle=True)
    ##############################################
    #Save the losses and accuracies to plot them together
    losses_list.append(history.history['loss'])
    val_losses_list.append(history.history['val_loss'])
    accuracies_list.append(history.history['accuracy'])
    val_accuracies_list.append(history.history['val_accuracy'])

#Naming the models to plot
model_names = ["MLP0","MLP1","MLP2","MLP3","MLP4","MLP5","MLP6","MLP7","MLP8","MLP9","MLP10","MLP11","MLP12","MLP13","MLP14","MLP15","MLP16","MLP17","MLP18","MLP19"]


#Plot for the losses
# Create a line plot for each model
plt.figure(figsize=(10, 6))
for i, name in enumerate(model_names):
    plt.plot(losses_list[i], label=name, marker='x')
    plt.plot(val_losses_list[i], label=name, marker='v',linestyle='--')

# Customize the plot
plt.title("Learning Curves of 20 best simpler NN Models(Loss)")
plt.xlabel("Epochs")
plt.ylabel("Loss(MSE)")
plt.legend(loc='upper right',fontsize=5.6)
plt.grid(True)
#plt.show()
plt.savefig('CUP/NN/Plots_all_models/loss_simpler_models20.png')
plt.clf

#################################################################

#Plot for the accuracies
# Create a line plot for each model
plt.figure(figsize=(10, 6))
for i, name in enumerate(model_names):
    plt.plot(accuracies_list[i], label=name, marker='x')
    plt.plot(val_accuracies_list[i], label=name, marker='v',linestyle='--')

# Customize the plot
plt.title("Learning Curves of 20 best simpler NN Models(Accuracy)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='upper right',fontsize=5.6)
plt.grid(True)
#plt.show()
plt.savefig(f'CUP/NN/Plots_all_models/acc_simpler_models20.png')
plt.clf