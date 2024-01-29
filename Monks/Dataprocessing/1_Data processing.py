#In this file we import each of the monks data sets to transform them into comma separated files, to then import them as data frames and work with them.
#Loading libraries
import pandas as pd

#Importing each test and transforming it
# Read the txt file into a DataFrame
for i in range(3):
    df = pd.read_table(f"Monks/Datasets/monks-{i+1}.test")
    # Replace spaces with commas in the 'col1' column
    # Assign new column names
    df.columns = [",class,a1,a2,a3,a4,a5,a6,ID"]
    df[",class,a1,a2,a3,a4,a5,a6,ID"] = df[",class,a1,a2,a3,a4,a5,a6,ID"].str.replace(" ", ",")

    # Save the DataFrame to a CSV file
    df.to_csv(f"Monks/Datasets/monks-{i+1}test_.csv")

#Importing each training and transforming it
# Read the txt file into a DataFrame
for i in range(3):
    df = pd.read_table(f"Monks/Datasets/monks-{i+1}.train")
    # Replace spaces with commas in the 'col1' column
    # Assign new column names
    df.columns = [",class,a1,a2,a3,a4,a5,a6,ID"]
    df[",class,a1,a2,a3,a4,a5,a6,ID"] = df[",class,a1,a2,a3,a4,a5,a6,ID"].str.replace(" ", ",")

    # Save the DataFrame to a CSV file
    df.to_csv(f"Monks/Datasets/monks-{i+1}train_.csv")


