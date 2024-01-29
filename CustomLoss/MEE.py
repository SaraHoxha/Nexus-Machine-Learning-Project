import tensorflow as tf

#For all the models, can be calculated in the end when we already have the predictions for all the models, only to fill the report tables.
def MEE(y_true, y_pred):     
   squared_error = (y_pred['X'] - y_true['X'])**2 + (y_pred['Y'] - y_true['Y'])**2 + (y_pred['Z'] - y_true['Z'])**2    
   mean_euclidean_error = tf.reduce_mean(tf.sqrt(squared_error))     
   return mean_euclidean_error