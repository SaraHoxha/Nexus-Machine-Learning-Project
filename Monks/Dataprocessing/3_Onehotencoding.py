#In this code we one hot encode all the datasets from the monks problems, this is to have 17 binary attributes instead of multiclass attributes
#Loading libraries
import pandas as pd

#Applying the one hot encode to each data set and saving it again
list_names = ['1test','1train','2test','2train','3test','3train']
for i in list(list_names):
    data = pd.read_csv(f'monk+s+problems/monks-{i}.csv')
    df = pd.DataFrame(data)

    # One-hot encode all categorical attributes
    encoded_df = pd.get_dummies(df, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
    encoded_df.to_csv(f"monk+s+problems/encoded_monks-{i}.csv")