#This file will visualize the true labels against the predicted labels in the 3d space, this is to check how far is the prediction grahically to the ground truth
#We want to see if the model is fitting nois or the general shape of the targets.

#Load libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#Defining the plotting function
def Plot_Valid_Pred(pred_values, val_val,path_to_save,name_to_save):
 
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for predicted values 
    ax.scatter(pred_values["X"], pred_values["Y"], pred_values["Z"], c='r', marker='o', label='Predicted')

    # Scatter plot for validation values 
    ax.scatter(val_val["X"], val_val["Y"], val_val["Z"], c='b', marker='o', label='Validation')

    # Set labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Set title
    ax.set_title('3D Scatter Plot')

    # Add legend
    ax.legend()

    plt.savefig(f'{path_to_save}/{name_to_save}.png')
    #plt.show()
    plt.clf()
