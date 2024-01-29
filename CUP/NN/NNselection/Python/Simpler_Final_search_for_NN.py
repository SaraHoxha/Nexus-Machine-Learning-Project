#In this file we decide the final structures of the models that are going to be used in the CUP, the decision was to split the search into 'complex' models and 'simpler' models, 'complex' that contain a higher number of layers, more options for the optimizer, higher number of units and higher number of regularization hyperparameters.
#The search finds the models with the best learning metrics and save them in to .json and also .keras files, these are then used to evaluate the final models.
#The file also creates a trials folder where it analizes the amount of trials we set, in our specific example, we analyzed 1000 models for the 'complex' case and 2000 for the 'simpler' case. then we save the 1% of the best models, we save, 10 models from the 'complex' case and 20 models from the 'simpler' case, in this way we mittigate the final decision to only the 1% found by the random search.

#This is a grid search algo based on the random search of keras, I use only the hyperparams choice in order to test specific values on each hyperparam.
#For this simpler search I changed the max num of layers to 4, and the max num of units per layer to 50 I also removed the relu and the silu activation functions, and for optimizers I only leave SGD or adam, as they are the most commonly used.
#Also increased the trials to 2000 and the number of epochs to 100.

#Load the libraries
import tensorflow as tf
import kerastuner
import keras
from keras import layers
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import json
import sys

#Import the data
data = pd.read_csv("Data_processing/train70.csv")
# Convert data to a NumPy array
data_array = data.to_numpy()
#Data process
#############################
# Get the first ten columns without id
inputs_training = data_array[:, 1:-3]
inputs_training.shape
# Get the last three columns
targets_training = data_array[:, -3:]
targets_training.shape
# using the train test split function 
x_train, x_test,y_train, y_test = train_test_split(inputs_training,targets_training, 
                                   random_state=104,  
                                   test_size=0.40,  
                                   shuffle=True) 
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

#Casting the data as tensors
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#Set up the number of trials for the grid search(Needs to be higher than the number of best models)
trials_to_use = 2000 #CHANGE THE TRIALS

def build_model(hp):
    #The grid with the final decision on hyperparameters
    dnn_layers_ss = [0,1,2]
    dnn_units_min, dnn_units_max = 1, 50
    dropout_ss = [0.1, 0.15, 0.2, 0.25]
    active_func_ss = ['sigmoid', 'tanh','leaky_relu','linear']
    optimizer_ss = ['SGD','adam']
    lr_min, lr_max = 1e-7, 1e-2
    momentum_min, momentum_max = 0,0.99
    weight_decay_min, weight_decay_max = 0,0.99
    rho_min, rho_max = 0,0.99
    epsilon_min, epsilon_max = 0,0.99
    i_a_min, i_a_max = 0,0.99

    #Passing the values to the hp functions
    active_func = hp.Choice('activation', active_func_ss)
    active_func2 = hp.Choice('activation2', active_func_ss)
    active_func3 = hp.Choice('activation3', active_func_ss)
    optimizer = hp.Choice('optimizer', optimizer_ss)
    lr = hp.Float('learning rate', min_value=lr_min, max_value=lr_max, sampling='log')
    momentum_val = hp.Float('momentum',min_value=momentum_min,max_value=momentum_max,sampling='linear')
    weight_decay_val = hp.Float('weight decay', min_value=weight_decay_min, max_value=weight_decay_max, sampling='linear')
    rho_val = hp.Float('rho', min_value=rho_min, max_value=rho_max, sampling='linear')
    epsilon_val = hp.Float('epsilon', min_value=epsilon_min, max_value=epsilon_max, sampling='linear')
    i_a_val = hp.Float('initial accumulator', min_value=i_a_min, max_value=i_a_max, sampling='linear')
    #dropout_val = hp.Choice('dropout rate',dropout_ss)
    #dropout_val2 = hp.Choice('dropout rate2',dropout_ss)

    ############################
    inputs = keras.Input(shape=(10,))
    #Creating the NN Architecture
    dnn_units = hp.Int(f"0_units", min_value=dnn_units_min, max_value=dnn_units_max)
    #first connection
    dense = keras.layers.Dense(units=dnn_units, activation=active_func)
    dense = dense(inputs)
    #hidden layers1
    for layer_i in range(hp.Choice("n_layers", dnn_layers_ss) - 1):
        dnn_units_alt = hp.Int(f"{layer_i}_1_units", min_value=dnn_units_min, max_value=dnn_units_max)
        dense = keras.layers.Dense(units=dnn_units_alt, activation=active_func2)(dense)
        if hp.Boolean("dropout"):
            dropout_val = hp.Choice('dropout rate',dropout_ss)
            dense = keras.layers.Dropout(rate=dropout_val)(dense)
    #Next hidden layers2
    for layer_i in range(hp.Choice("n_layers2_", dnn_layers_ss) - 1):
        dnn_units_alt2 = hp.Int(f"{layer_i}_2_units", min_value=dnn_units_min, max_value=dnn_units_max)
        dense = keras.layers.Dense(units=dnn_units_alt2, activation=active_func3)(dense)
        if hp.Boolean("dropout"):
            dropout_val2 = hp.Choice('dropout rate2',dropout_ss)
            dense = keras.layers.Dropout(rate=dropout_val2)(dense)
    #Output layer
    outputs = layers.Dense(3)(dense)
    #Model creation
    model = keras.Model(inputs=inputs, outputs=outputs)
    #The desired optimizer
    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=lr,momentum=momentum_val,weight_decay=weight_decay_val)
    elif optimizer == "RMSprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=lr,rho=rho_val,momentum=momentum_val,epsilon=epsilon_val,weight_decay=weight_decay_val)
    elif optimizer == "adagrad":
        optimizer = keras.optimizers.Adagrad(learning_rate=lr,initial_accumulator_value=i_a_val,epsilon=epsilon_val,weight_decay=weight_decay_val)
    else:
        raise("Not supported optimizer")
    # compile model
    model.compile(optimizer=optimizer,
                  loss=keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    return model

#Building the tuner with different techniques
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

#Search parameter    
obj = kerastuner.Objective('val_accuracy', direction='max')
dir_name = "CUP/NN/NNselection/simpler/Final_search"
randomsearch_tuner = build_tuner(build_model, "RandomSearch", obj, dir_name)
randomsearch_tuner.search(x_train,y_train,
             epochs=100,#EPOCHS TO USE IN THE GRIDSEARCH
             validation_data=(x_test,y_test),verbose=0)


# Capture the terminal output
old_stdout = sys.stdout

# Create a file object to write to
file_object = open("CUP/NN/NNselection/simpler/terminal.txt", "w")

# Set sys.stdout to the file object
sys.stdout = file_object

#Let's take the 10 best performing models and evaluate them
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

for i in range(20):
    print('===================================================================')
    print(f'Model_{i}_best_hyperparams')
    best_hyperparameters = randomsearch_tuner.get_best_hyperparameters(20)[i]
    # Write the JSON object to a file
    with open(f'CUP/NN/NNselection/simpler/Hyperparams/json/model_{i}.json', 'w') as f:
        json.dump(best_hyperparameters.values, f)
    print(best_hyperparameters.values)
     
#Showing and saving the N best models obtained from the search
n_best_models = randomsearch_tuner.get_best_models(num_models=20)
for i in range(20):
    model_structure = n_best_models[i].to_json()
    # Write the JSON object to a file
    with open(f'CUP/NN/NNselection/simpler/Models/json/model_{i}.json', 'w') as f:
        json.dump(model_structure, f)
    print(n_best_models[i].summary())                    # best-model summary

for i in range(20):
    best_hyperparameters = randomsearch_tuner.get_best_hyperparameters(20)[i]
    print(f'Model{i}_best_performing')
    for train, test in kfold.split(inputs_training, targets_training):
        model = randomsearch_tuner.hypermodel.build(best_hyperparameters)
        # Calling `save('my_model.keras')` creates a zip archive `my_model.keras`.
        model.save(f'CUP/NN/NNselection/simpler/Models/keras/model_{i}.keras')
        history = model.fit(inputs_training[train],targets_training[train],
                            epochs=100, #EPOCHS FOR THE BEST 10 MODELS AND THEIR RESPECTIVE CROSS VAL
                            validation_data=(inputs_training[test], targets_training[test]),verbose=0)
        ###############################################################
        #Final values on test data
        test_scores = model.evaluate(inputs_training[test], targets_training[test], verbose=0)
        ###############################################################
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
    ##############################
    # evaluate model
    #_, acc = model.evaluate(x_test, y_test, verbose=0)
    #print('> %.3f' % (acc * 100.0))
    ##############################
    
    #Reset the kfold for the next model
    fold_no = 0
    acc_per_fold = []
    loss_per_fold = []
print('DONE')

# Restore sys.stdout
sys.stdout = old_stdout

# Close the file object
file_object.close()

