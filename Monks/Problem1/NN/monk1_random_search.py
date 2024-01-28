#This is a grid search algo based on the random search of keras, I use only the hyperparams choice in order to test specific values on each hyperparam.
#For this simpler search I changed the max num of layers to 1, and the max num of units per layer to 11, I only leave SGD or adam, as they are the most commonly used.
#Also increased the trials to 500 and the number of epochs to 300.
#The loss function is changed to have the cross binary entropy for classification problems
#The output unit got a sigmoid activation function in order to obtain probabilities of 0 and 1
#Load libraries
import tensorflow as tf
import kerastuner
import keras
from keras import layers
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
import pandas as pd
import numpy as np
import json
#Import the datasets
data = pd.read_csv("C:/Users/urbi1/OneDrive/Escritorio/ML_2023/monk+s+problems/encoded_monks-1train.csv")
data_test = pd.read_csv("C:/Users/urbi1/OneDrive/Escritorio/ML_2023/monk+s+problems/encoded_monks-1test.csv")
# Convert data to a NumPy array
data_array = data.to_numpy()
data_test_array = data_test.to_numpy()
#Data process
#############################
# Get the 17 encoded attributes
inputs_training = data_array[:, 3:]
inputs_training.shape
inputs_training = np.asarray(inputs_training).astype('int')
# Get the target class
targets_training = data_array[:,1]
targets_training.shape
targets_training = np.asarray(targets_training).astype('int')
# Get the 17 encoded attributes
inputs_test = data_test_array[:, 3:]
inputs_test.shape
inputs_test = np.asarray(inputs_test).astype('int')
# Get the target class
targets_test = data_test_array[:,1]
targets_test.shape
targets_test = np.asarray(targets_test).astype('int')
#Cast the data as int in order to receive 0's and 1's
x_train = np.asarray(inputs_training).astype('int')
x_test = np.asarray(inputs_test).astype('int')
y_train = np.asarray(targets_training).astype('int')
y_test = np.asarray(targets_test).astype('int')

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#Set up the number of trials for the grid search(Needs to be higher than the number of best models)
trials_to_use = 500 #CHANGE THE TRIALS

def build_model(hp):
    #The grid of hyperparams
    dnn_layers_ss = [1]
    dnn_units_min, dnn_units_max = 4,11
    active_func_ss = ['sigmoid', 'tanh']
    optimizer_ss = ['SGD','adam']
    lr_min, lr_max = 1e-2, 0.75
    momentum_min, momentum_max = 0.05,0.9
    weight_decay_min, weight_decay_max = 0.05,0.9

    #Passing the values to the hp functions(To create the combinations)
    active_func = hp.Choice('activation', active_func_ss)
    active_func2 = hp.Choice('activation2', active_func_ss)
    optimizer = hp.Choice('optimizer', optimizer_ss)
    lr = hp.Float('learning rate', min_value=lr_min, max_value=lr_max, sampling='linear')
    momentum_val = hp.Float('momentum',min_value=momentum_min,max_value=momentum_max,sampling='linear')
    weight_decay_val = hp.Float('weight decay', min_value=weight_decay_min, max_value=weight_decay_max, sampling='linear')

    #Neural Network architecture
    ############################
    inputs = keras.Input(shape=(17,))
    #Creating the NN Architecture
    dnn_units = hp.Int(f"0_units", min_value=dnn_units_min, max_value=dnn_units_max)
    #first layer connection
    dense = keras.layers.Dense(units=dnn_units, activation=active_func,kernel_initializer=keras.initializers.RandomNormal(mean=0.001,stddev=0.001),
    bias_initializer=keras.initializers.Zeros())
    dense = dense(inputs)
    #Next hidden layers1
    for layer_i in range(hp.Choice("n_layers", dnn_layers_ss) - 1):
        dnn_units_alt = hp.Int(f"{layer_i}_1_units", min_value=dnn_units_min, max_value=dnn_units_max)
        dense = keras.layers.Dense(units=dnn_units_alt, activation=active_func2,kernel_initializer=keras.initializers.RandomNormal(mean=0.001,stddev=0.001),
    bias_initializer=keras.initializers.Zeros())(dense)
    #Output layer
    outputs = layers.Dense(1,activation='sigmoid')(dense)
    model = keras.Model(inputs=inputs, outputs=outputs)
    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=lr,momentum=momentum_val,weight_decay=weight_decay_val)
    else:
        raise("Not supported optimizer")
    # compile model
    model.compile(optimizer=optimizer,
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model
#Tuner for the search
def build_tuner(model, hpo_method, objective, dir_name):
    if hpo_method == "RandomSearch":
        tuner = RandomSearch(model, objective=objective, max_trials=trials_to_use, executions_per_trial=1,
                               project_name=hpo_method, directory=dir_name)
    elif hpo_method == "Hyperband":
        tuner = Hyperband(model, objective=objective, max_epochs=3, executions_per_trial=1,
                            project_name=hpo_method)
    elif hpo_method == "BayesianOptimization":
        tuner = BayesianOptimization(model, objective=objective, max_trials=trials_to_use, executions_per_trial=1,
                                       project_name=hpo_method)
    return tuner
#Params for the tuner  
obj = kerastuner.Objective('val_accuracy', direction='max')
dir_name = "monk+s+problems/Problem1/Trials"
randomsearch_tuner = build_tuner(build_model, "RandomSearch", obj, dir_name)
randomsearch_tuner.search(x_train,y_train,
             epochs=300,#EPOCHS TO USE IN THE GRIDSEARCH
             validation_data=(x_test,y_test),verbose=0)
#Showing and evaluating the best models
for i in range(5):
    print('===================================================================')
    print(f'Model_{i}_best_hyperparams')
    best_hyperparameters = randomsearch_tuner.get_best_hyperparameters(5)[i]
    # Write the JSON object to a file
    with open(f'monk+s+problems/Problem1/Hyperparams/json/model_{i}.json', 'w') as f:
        json.dump(best_hyperparameters.values, f)
    print(best_hyperparameters.values)
#Showing and saving the N best models obtained from the search
n_best_models = randomsearch_tuner.get_best_models(num_models=5)
for i in range(5):
    model_structure = n_best_models[i].to_json()
    # Write the JSON object to a file
    with open(f'monk+s+problems/Problem1/Models/json/model_{i}.json', 'w') as f:
        json.dump(model_structure, f)
    print(n_best_models[i].summary())                    # best-model summary
#Showing and saving the N best models obtained from the search
for i in range(5):
    best_hyperparameters = randomsearch_tuner.get_best_hyperparameters(5)[i]
    model = randomsearch_tuner.hypermodel.build(best_hyperparameters)
    model.save(f'monk+s+problems/Problem1/Models/keras/model_{i}.keras')
    print(f'Model{i}_best_performing')
    
print('DONE')


