#This code will focus on showing the correlations between the input attributes and the targets, this will enable us to have a better understanding of what attributes are important and which ones are not.
#Load libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Import datasets
# Read the CSV file into a Pandas DataFrame
df_training = pd.read_csv("ML-CUP23-TR.csv")
df_test = pd.read_csv("ML-CUP23-TS.csv")

# Create the correlation heatmap excluding the 'id' column
columns_to_include = df_training.columns.difference(['id'])

# Display the correlation heatmap for the selected attributes
plt.figure(figsize=(10, 8)) 
sns.heatmap(df_training[columns_to_include].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()