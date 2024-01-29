#Brief understanding of the categories and the data in general, how are the frequencies for different categories
#Loading libraries
import pandas as pd
import matplotlib.pyplot as plt
#Importing the data set
df = pd.read_csv('Monks/Datasets/monks-1train.csv')
#Checking the value counts for each category
unique_values_counts = df['a1'].value_counts()
unique_values_counts_dict = unique_values_counts.to_dict()

categories = list(unique_values_counts_dict.keys())
values = list(unique_values_counts_dict.values())

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(categories,values, color=['b', 'g', 'r', 'c'])

# Add labels and title
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.title('Barchart for attribute')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Customize grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the chart
plt.tight_layout()
plt.show()