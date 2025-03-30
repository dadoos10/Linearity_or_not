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
from toolBox import *

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

def handle_duplicates_and_sort(data_1, data_2):
    """
    This function handles duplicates and sorts the data by Lipid fraction values.
    :return: None
    """
    print(f"data_1: \n{data_1}.\n data_2: \n{data_2}")
    # delete duplicates of Lipid fraction values, and keep the first one.
    data_1 = data_1.drop_duplicates(subset=['Lipid (fraction)'], keep='first')
    data_2 = data_2.drop_duplicates(subset=['Lipid (fraction)'], keep='first')
    # order datasets by Lipid fraction values.
    data_1 = data_1.sort_values(by=['Lipid (fraction)'])
    data_2 = data_2.sort_values(by=['Lipid (fraction)'])
    return data_1, data_2

def plot_linearity(data_1, data_2, exp1, exp2, mri_param = 'R1 (1/sec)',lipid = True):
    to_show = {True: 'Lipid type', False: 'Iron type'}
    # # Create a scatter plot for the data, the x axis is the R1 (1/sec) of data_1, and the y axis is the R1 (1/sec) of data_12.
    # # The title of the plot is "R1 (1/sec) of data_1 vs. R1 (1/sec) of data_12"
    # # The x axis label is "R1 (1/sec) of data_1" and the y axis label is "R1 (1/sec) of data_12".
    plt.figure(figsize=(10, 6))
    plt.scatter(data_1[mri_param], data_2[mri_param], alpha=0.5)
    plt.title(f'{mri_param} of {exp1} vs. {mri_param} of {exp2} - for {data_1[to_show[lipid]].iloc[0]}')
    plt.xlabel(f'{mri_param} of {exp1}')
    plt.ylabel(f'{mri_param} of {exp2}')
    plt.grid(True)
    plt.show()

def run_test_retest(data,exps_pair,MRI_param = 'R1 (1/sec)',lipid = True):
    data_1 =  extract_zero_com_exp(exps_pair[0], data, lipid)
    data_2 =  extract_zero_com_exp(exps_pair[1], data, lipid)
    data_1,data_2 = handle_duplicates_and_sort(data_1, data_2)
    plot_linearity(data_1, data_2, exps_pair[0], exps_pair[1],MRI_param,lipid)


# main function
if __name__ == "__main__":
    print("d")
    # # Read the data file into a pandas dataframe
    data = pd.read_excel('data.xlsx', sheet_name=0)
    # exp_lipid_pairs_to_check = [PC_Cholest_pair,PC_SM_pair,PC_pair]
    # for lipid_pair in exp_lipid_pairs_to_check:
    #     run_test_retest(data,lipid_pair)

    exp_iron_pairs_to_check = [Fe2_pair, Fe3_pair, Ferittin_pair, Tranferrin_pair]
    for iron_pair in exp_iron_pairs_to_check:
        run_test_retest(data,iron_pair, lipid = False)



    


