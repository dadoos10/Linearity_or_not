# Rona's Project:
# In this project we research the linearity relationship between biological components to quantitative MRI (qMRI).
# it is well known (source!) that lypid and iron represent linear connection to qMRI parameters, seperatly. 
# Here we want to ask if the combination of the two types represent linearity as well.
# the players: 
# biological components: lipids: PC,PC_SphingoMyalin.....
# qMRI parameters: R1,R2s,MT,MTV.  

# first we want to crate a scan rescan plot, to see the stability of our experimants.
# Actualy we make two plots, one for iron, one for lipyd, since they behave different.
# now we open the data file, and read it into a pandas dataframe.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from toolBox import *
from sklearn.metrics import mean_squared_error


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




def group_and_mean(data_1, data_2, column_name):
    # Group by ExpNum and column_name, calculating the mean for only the qMRI_params columns
    data_1_qMRI = data_1.groupby(["ExpNum", column_name], as_index=False)[qMRI_params].mean(numeric_only=True)
    data_2_qMRI = data_2.groupby(["ExpNum", column_name], as_index=False)[qMRI_params].mean(numeric_only=True)

    # Keep non-numeric columns (if any), retaining the first value for each group (or you can use .last() instead)
    non_qMRI_columns = [col for col in data_1.columns if col not in qMRI_params]

    # Group by ExpNum and column_name, and take the first value for non-numeric columns
    data_1_non_qMRI = data_1.groupby(["ExpNum", column_name], as_index=False)[non_qMRI_columns].first()
    data_2_non_qMRI = data_2.groupby(["ExpNum", column_name], as_index=False)[non_qMRI_columns].first()

    # Merge the mean qMRI_params with the non-numeric columns
    data_1_final = pd.merge(data_1_qMRI, data_1_non_qMRI, on=["ExpNum", column_name])
    data_2_final = pd.merge(data_2_qMRI, data_2_non_qMRI, on=["ExpNum", column_name])
    return data_1_final, data_2_final

def handle_duplicates_and_sort(data_1, data_2, lipid=True):
    """
    Handles duplicates and sorts data by Lipid fraction or Fe values.
    
    :param data_1: First DataFrame
    :param data_2: Second DataFrame
    :param lipid: Boolean flag to choose column (True for Lipid fraction, False for Fe)
    :return: Processed data_1 and data_2
    """
    subset = {True: ['Lipid (fraction)'], False: ['[Fe] (mg/ml)']}
    column_name = subset[lipid][0]  # Extract the correct column name
    

    data_1, data_2 = group_and_mean(data_1, data_2, column_name)

    # # Ensure both datasets have rows in common based on the selected column
    # common_values = pd.merge(data_1, data_2, on=column_name, how='inner')[column_name]

    # # Filter to keep only common values
    # data_1 = data_1[data_1[column_name].isin(common_values)]
    # data_2 = data_2[data_2[column_name].isin(common_values)]

    # print(f"Filtered data_1: \n{data_1}.\nFiltered data_2: \n{data_2}")

    # Sort datasets by the selected column
    data_1 = data_1.sort_values(by=column_name)
    data_2 = data_2.sort_values(by=column_name)

    return data_1, data_2

# def plot_linearity(data_1, data_2, exp1, exp2, mri_param = 'R1 (1/sec)',lipid = True):
    to_show = {True: 'Lipid type', False: 'Iron type'}
        # create plot directory
    plot_dir = 'plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if lipid:
        # create lipid directory
        if not os.path.exists(f'{plot_dir}/lipid'):
            os.makedirs(f'{plot_dir}/lipid')
        plot_dir = f'{plot_dir}/lipid'
    else:
        # create iron directory
        if not os.path.exists(f'{plot_dir}/iron'):
            os.makedirs(f'{plot_dir}/iron')
        plot_dir = f'{plot_dir}/iron'
    # save plot
    filename = f"{mri_param}_{exp1}_vs_{exp2}.png".replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
    # # Create a scatter plot for the data, the x axis is the R1 (1/sec) of data_1, and the y axis is the R1 (1/sec) of data_12.
    # # The title of the plot is "R1 (1/sec) of data_1 vs. R1 (1/sec) of data_12"
    # # The x axis label is "R1 (1/sec) of data_1" and the y axis label is "R1 (1/sec) of data_12".
    plt.figure(figsize=(10, 6))
    plt.scatter(data_1[mri_param], data_2[mri_param], alpha=0.5)
    plt.title(f'{mri_param} of {exp1} vs. {mri_param} of {exp2} - for {data_1[to_show[lipid]].iloc[0]}')
    plt.xlabel(f'{mri_param} of {exp1}')
    plt.ylabel(f'{mri_param} of {exp2}')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, filename))
    # plt.show()
    plt.close()

def make_datasets_same_size(data, data_2):

    return data, data_2

def calc_scan_rescan_rmse(data, data_2, R1):
    if len(data) != len(data_2):
        data, data_2 = make_datasets_same_size(data, data_2)
        return -10
    # Compute RMSEs
    scan_rescan_rmse = np.sqrt(np.mean((data[R1].values - data_2[R1].values) ** 2))
    return scan_rescan_rmse

def plot_qMRI_to_bio(data, data_2, R1, exps_pair,i,lipid = True):
    if lipid:
        tissue_col = "Lipid (fraction)"
        tissue_type = data["Lipid type"].iloc[0]
    else:
        tissue_col = "[Fe] (mg/ml)"
        tissue_type = data["Iron type"].iloc[0]
    # Extract x and y values (drop NaNs to avoid issues)
    x = data[tissue_col].dropna()
    y = data[R1].dropna()

    # Ensure matched lengths
    x, y = x.align(y, join='inner')

    # Perform linear regression
    coefficients = np.polyfit(x, y, 1)
    linear_fit = np.poly1d(coefficients)

    # Prepare scatter and regression line
    scatter1 = plt.scatter(x, y, alpha=0.5, color='blue')  # Data 1
    scatter2 = plt.scatter(data_2[tissue_col], data_2[R1], alpha=0.5, color='green')  # Data 2
    line = plt.plot(x, linear_fit(x), color='red')  # Regression line
    print(f"{tissue_type} ################################")
    # Add legend manually
    others = [x for x in range(len(exps_pair)) if x != i]
    # Create a string of experiment numbers from `others`, joined by commas
    others_str = ', '.join([f"exp # {exps_pair[x]}" for x in others])
    plt.legend([scatter1, scatter2, line[0]], [f"{others_str}", f"exp # {exps_pair[i]}", f"Linear fit of {data["ExpNum"].iloc[0]}"])

    # Labels and title
    plt.xlabel(tissue_col)
    plt.ylabel(R1)


    # Calculate RMSEs
    scan_rescan_rmse = calc_scan_rescan_rmse(data, data_2, R1)

    y_on_line = linear_fit(data_2[tissue_col])  # The "ideal" y values on the line
    y_actual = data_2[R1]
    print(f"{R1}, {y_actual}, {y_on_line}, tissue_col: {tissue_col}")
    if len(y_actual) == 0 or len(y_on_line) == 0:
        rmse_fitted = -10
    else:
        rmse_fitted = np.sqrt(mean_squared_error(y_actual, y_on_line))

    # Print RMSEs
    print(f'RMSE for the fitted line (data_1): {rmse_fitted:.4f}')
    # print(f'Scan–rescan RMSE: {scan_rescan_rmse:.4f}')

    param_name = R1.split(" ")[0]
    # Final plot details
    plt.title(f'{param_name} vs. {tissue_col}, {tissue_type}\n RMSE (fit): {rmse_fitted:.2f}')
    plt.grid(True)

    to_show = {True: 'Lipid type', False: 'Iron type'}
    if lipid:
        plot_dir = f'plots/lipid/{param_name}'
    else:
        plot_dir = f'plots/iron/{param_name}'
    filename = f"{param_name}_pure {tissue_type}.png".replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
    save_file(plot_dir, filename)
    return rmse_fitted, scan_rescan_rmse
    
def create_nested_dir(dir_path):
    """Create nested directories one by one if they don't exist."""
    current_path = ""
    for folder in dir_path.split("/"):
        if not folder:  # Skip empty parts (e.g., if path starts with "/")
            continue
        current_path = os.path.join(current_path, folder)
        if not os.path.exists(current_path):
            os.makedirs(current_path)
    return current_path

def save_file(plot_dir, filename):
    """
    Save the plot to the specified directory with the given filename.
    """
    plot_dir = create_nested_dir(plot_dir)
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()


def run_test_retest(data,exps_pair,MRI_param = 'R1 (1/sec)',lipid = True):
    rmse_table = pd.DataFrame(
        data=0.0, 
        index=qMRI_params, 
        columns=["Fitted RMSE", "Scan-Rescan RMSE"]
    )
    for i in range(len(exps_pair)):
        # define data_2 as the i experiment

        data_2 = extract_zero_com_exp(exps_pair[i], data, lipid)
        # create data_1 from all other experiments except i
        data_1_list = [
            extract_zero_com_exp(exps_pair[j], data, lipid)
            for j in range(len(exps_pair)) if i != j
        ]
        data_1 = pd.concat(data_1_list, ignore_index=True)

        data_1,data_2 = handle_duplicates_and_sort(data_1, data_2,lipid)
        for param in qMRI_params:
            rmse_fitted, scan_rescan_rmse =  plot_qMRI_to_bio(data_1, data_2, param, exps_pair,i, lipid = lipid)
            rmse_table.loc[param, "Fitted RMSE"] += rmse_fitted
            # rmse_table.loc[param, "Scan-Rescan RMSE"] += scan_rescan_rmse
            print(rmse_table)

    rmse_table /= len(exps_pair)
    rmse_table = rmse_table.round(3)

    print(rmse_table)

    # for data_2 plot the R1 (1/sec) of data_2 vs. MTV (fraction)
    # plot_linearity(data_1, data_2, exps_pair[0], exps_pair[1],MRI_param,lipid = lipid)


# main function
if __name__ == "__main__":
    # # Read the data file into a pandas dataframe
    data = pd.read_excel('data.xlsx', sheet_name=0)
    exp_lipid_all_to_check  = [PC_all,PC_Cholest_all,PC_SM_all]
    exp_iron_all_to_check = [Fe2_all, Fe3_all, Ferittin_all, Tranferrin_all]


    exp_lipid_pairs_to_check = [PC_Cholest_pair,PC_SM_pair,PC_pair]
    exp_iron_pairs_to_check = [Fe2_pair, Fe3_pair, Ferittin_pair, Tranferrin_pair]
    for iron_pair in exp_iron_all_to_check:
        run_test_retest(data,iron_pair, lipid = False)

    for lipid_pair in exp_lipid_all_to_check:
        run_test_retest(data,lipid_pair)



    


