#This file is only for creating the custom loss that professor Micheli asked in the guidelines
import tensorflow as tf

#For all the models, can be calculated in the end when we already have the predictions for all the models, only to fill the report tables.
#You pass the dataframes containing the predicted and true labels in X,Y,Z
def MEE(y_true, y_pred):
   euclidean_error = tf.reduce_sum((y_pred['X'] - y_true['X'])**2+(y_pred['Y'] - y_true['Y'])**2+(y_pred['Z'] - y_true['Z'])**2)**(1/2)
   # Calculate the mean of the Euclidean distances
   mean_euclidean_error = tf.reduce_mean(euclidean_error)
   return mean_euclidean_error