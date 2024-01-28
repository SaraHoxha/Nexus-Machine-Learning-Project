#In this file we do an initial exploration on the data to check potential behaviors, correlations, and errors on the data
#Load libraries
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
#Import datasets
# Read the CSV file into a Pandas DataFrame
df_training = pd.read_csv("ML-CUP23-TR.csv")
df_test = pd.read_csv("ML-CUP23-TS.csv")

# Extract the values from the DataFrame as a NumPy array
data_training = df_training.values
data_test = df_test.values

# Convert the NumPy array to a PyTorch tensor
training_tensor = torch.from_numpy(data_training)
test_tensor = torch.from_numpy(data_test)
#The shape of the tensors
training_tensor.shape
test_tensor.shape

################################################################
#SAMPLING
#Remember that the test set does not contain x,y,z
#Check the num uniques
df_training.nunique()
df_test.nunique()

#Plot a histogram for each of the variables to check the distributions
for i in df_training.columns:
    if i == "id":
        continue
    print('var'+str(i))
    print(len(np.unique(df_training[i])))
    #Data sampling
    num_bins = int(np.sqrt(len(df_training[i])))
    plt.hist(df_training[i], bins=50, color='skyblue')  # Adjust the number of bins as needed
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram variable: '+str(i), fontsize=16, color='black')
    plt.savefig(f'DataUnderstanding/histogram_{i}.png')
    #plt.show()
    plt.clf()

################################################################
#TARGETS SAMPLING
# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_training["X"], df_training["Y"], df_training["Z"],color='blue', marker='x', label='label')
ax.set_xlabel('X Target')
ax.set_ylabel('Y Target')
ax.set_zlabel('Z Target')
ax.set_title('3D target visualization')
plt.savefig(f'DataUnderstanding/3D_Targets_plot.png')
#plt.show()
plt.clf()


################################################################
#Not so useful, but this is a visualization on all posible combinations from the inputs that we have
#They resemble a lot, and are highly concentrated in the extremes close to 1 or -1
columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
combinations = itertools.combinations(columns, 3)
for idx, (x, y, z) in enumerate(combinations, start=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_training[x], df_training[y], df_training[z], c='r', marker='o', label='label')
    plt.suptitle("Scatters for inputs: "+x+" "+y+" "+z)
    ax.set_xlabel(f'{x} Label')
    ax.set_ylabel(f'{y} Label')
    ax.set_zlabel(f'{z} Label')
    ax.set_title(f'Input variables {x}, {y}, {z}, plotted together')
    #plt.show()

################################################################
#WHAT ARE THE RANGES WHERE THE TARGETS ARE AND THE ONES FOR THE ATTRIBUTES?
#We can see that the inputs tend to be standardized close to -1 and 1, and the targets move more widely in the positive and negative ranges.
columns_to_plot = df_training.columns.difference(['id'])
df_transposed = df_training[columns_to_plot].T

# Plotting the heatmap
plt.figure(figsize=(14, 10))  # Adjust the figure size as needed
sns.heatmap(df_transposed, cmap='nipy_spectral')
plt.title('Title') #Title
plt.xlabel('Observations')
plt.ylabel('Variables')
plt.title('Heatmap of Attributes vs targets')
plt.legend(['Observations', 'Variables'], loc='upper left') 
#Adds a legend to the plot, indicating which line corresponds to what
plt.tight_layout()
#Prevent overlapping
plt.savefig(f'DataUnderstanding/Heatmap_of_all_obs.png')
#plt.show()
################################################################


