print("hi")
# Rona's Project:
# In this project we research the linearity relationship between biological components to quantitative MRI (qMRI).
# it is well known (source!) that lypid and iron represent linear connection to qMRI parameters, seperatly. 
# Here we want to ask if the combination of the two types represent linearity as well.
# the players: 
# biological components: lipids: PC,PC_SphingoMyalin.....
# qMRI parameters: R1,R2s,MT,MTV.  
# import pandas as pd
# import NumPy as np
# first we want to crate a scan rescan plot, to see the stability of our experimants.
# Actualy we make two plots, one for iron, one for lipyd, since they behave different.
# now we open the data file, and read it into a pandas dataframe.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_zero_com_exp(exp = 1, data = None, lipid = False):
    """
    This function extracts the data for a given experiment number and lipid type.
    :param exp: Experiment number (default is 1)
    :param data: DataFrame containing the data
    :param lipid: Boolean indicating if the data is for lipids (default is False)
    :return: DataFrame with the extracted data
    """
    if lipid:
        return data[(data['ExpNum'] == exp) & (data['[Fe] (mg/ml)'] == 0)]
    else:
        return data[(data['ExpNum'] == exp) & (data['Lipid (fraction)'] == 0)]

# # Read the data file into a pandas dataframe
data = pd.read_excel('data.xlsx', sheet_name=0)
exp = 1
data_1 =  extract_zero_com_exp(exp, data = data, lipid = True)
exp = 12
data_12 =  extract_zero_com_exp(exp, data = data, lipid = True)
print(data_1)
print(data_12)
# delete duplicates of Lipid fraction values, and keep the first one.
data_1 = data_1.drop_duplicates(subset=['Lipid (fraction)'], keep='first')
data_12 = data_12.drop_duplicates(subset=['Lipid (fraction)'], keep='first')
# # Create a scatter plot for the data, the x axis is the R1 (1/sec) of data_1, and the y axis is the R1 (1/sec) of data_12.
# # The title of the plot is "R1 (1/sec) of data_1 vs. R1 (1/sec) of data_12"
# # The x axis label is "R1 (1/sec) of data_1" and the y axis label is "R1 (1/sec) of data_12".
plt.figure(figsize=(10, 6))
plt.scatter(data_1['R1 (1/sec)'], data_12['R1 (1/sec)'], alpha=0.5)
plt.title('R1 (1/sec) of data_1 vs. R1 (1/sec) of data_12')
plt.xlabel('R1 (1/sec) of data_1')
plt.ylabel('R1 (1/sec) of data_12')
plt.grid(True)
plt.show()

