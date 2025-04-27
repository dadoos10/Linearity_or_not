# Rona's Project:
# On this project we research the linearity relationship between biological components to quantitative MRI (qMRI).
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
import pickle
from toolBox import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


def extract_zero_com_exp(exp = 1, data = None, lipid = False):
    """
    This function extracts the data for a given experiment number and lipid type.
    :param exp: Experiment number (default is 1)
    :param data: DataFrame containing the data
    :param lipid: Boolean indicating if the data is for lipids (default is False)
    :return: DataFrame with the extracted data
    """
    if lipid:
        #
        return data[(data['ExpNum'] == exp) & (data['[Fe] (mg/ml)'] == 0) & (data['[Protein](mg/ml)'] == 0)] 
    else:
        return data[(data['ExpNum'] == exp) & (data['Lipid (fraction)'] == 0) ]




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


    # Create new figure
    fig, ax = plt.subplots()

    # Prepare scatter and regression line
    scatter1 = ax.scatter(x, y, alpha=0.5, color='blue')  # Data 1
    scatter2 = ax.scatter(data_2[tissue_col], data_2[R1], alpha=0.5, color='green')  # Data 2
    line, = ax.plot(x, linear_fit(x), color='red')  # Regression line
    print(f"{tissue_type} ################################")
    # Add legend manually
    others = [x for x in range(len(exps_pair)) if x != i]
    # Create a string of experiment numbers from `others`, joined by commas
    others_str = ', '.join([f"exp # {exps_pair[x]}" for x in others])
    ax.legend([scatter1, scatter2, line], [f"{others_str}", f"exp # {exps_pair[i]}", f"Linear fit"])

    # Labels and title
    ax.set_xlabel(tissue_col)
    ax.set_ylabel(R1)


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
    ax.set_title(f'{param_name} vs. {tissue_col}, {tissue_type}\n RMSE (fit): {rmse_fitted:.2f}')
    ax.grid(True)
    return fig,ax,rmse_fitted, scan_rescan_rmse, (param_name, tissue_col, tissue_type)

def define_dir_and_save(lipid, param_name, tissue_type):
    if lipid:
        plot_dir = f'plots/lipid/{param_name}'
    else:
        plot_dir = f'plots/iron/{param_name}'
    filename = f"{param_name}_pure {tissue_type}.png".replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
    save_file(plot_dir, filename)

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
    fig_path = os.path.join(plot_dir, filename.replace('.png', '.fig.pickle'))
    with open(fig_path, 'wb') as f:
        pickle.dump(plt.gcf(), f)
    plt.close()


def run_scan_rescan(data,exps_pair,rmse_dict,lipid = True):
    rmse_table = pd.DataFrame(
        data={col: [[] for _ in range(len(qMRI_params))] for col in ["Fitted RMSE", "Scan-Rescan RMSE"]},
        index=qMRI_params
    )
    for param in qMRI_params:
        rmses = []
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
                
            fig,ax,rmse_fitted, scan_rescan_rmse, (param_name, tissue_col, tissue_type) =  plot_qMRI_to_bio(data_1, data_2, param, exps_pair,i, lipid = lipid)
            rmses.append(rmse_fitted)
            # if isinstance(rmse_table.loc[param, "Fitted RMSE"], list):
            #     rmse_table.loc[param, "Fitted RMSE"].append(rmse_fitted)
            # else:
            #     rmse_table.loc[param, "Fitted RMSE"] = [rmse_fitted]

        # mean_rmse_fitted = np.mean(rmse_table.loc[param, "Fitted RMSE"]) if rmse_table.loc[param, "Fitted RMSE"] else 0
        mean_rmse_fitted = np.mean(rmses) if rmses else 0
        ax.set_title(f'{param_name} vs. {tissue_col}, {tissue_type}\n average RMSE (fit): {mean_rmse_fitted:.2f}')
        define_dir_and_save(lipid, param_name, tissue_type)
        rmse_dict.setdefault(param, []).append((tissue_type, rmses))
            # rmse_table.loc[param, "Scan-Rescan RMSE"] += scan_rescan_rmse
            # print(rmse_table)
    return rmse_dict

def dict_to_boxplot(rmse_dict):
     # plot the RMSEes for each rmse_dict
    plt.close('all')
    plt.clf()

    for param, values in rmse_dict.items():
        #create a boxplot for each param, the title of the barplot is the param name, and the y axis is the RMSE values.
        
        plt.figure(figsize=(10, 6))
        plt.title(f'RMSE for {param}')
        plt.boxplot([x[1] for x in values], tick_labels=[x[0] for x in values])
        plt.ylabel('RMSE')
        plt.xlabel('Tissue type')
        plt.grid(True)
        plt.show()
        plt.close()

def pure_components(data):
    exp_lipid_all_to_check  = [PC_all,PC_Cholest_all,PC_SM_all]
    exp_iron_all_to_check = [Fe2_all, Fe3_all, Ferittin_all, Tranferrin_all]


    exp_lipid_pairs_to_check = [PC_Cholest_pair,PC_SM_pair,PC_pair]
    exp_iron_pairs_to_check = [Fe2_pair, Fe3_pair, Ferittin_pair, Tranferrin_pair]
    rmse_dict = {}
    for iron_pair in exp_iron_all_to_check:
        rmse_dict = run_scan_rescan(data,iron_pair, rmse_dict,lipid = False)
    
    dict_to_boxplot(rmse_dict)
    rmse_dict = {}
   
    for lipid_pair in exp_lipid_all_to_check:
        rmse_dict = run_scan_rescan(data,lipid_pair,rmse_dict)

    dict_to_boxplot(rmse_dict)

def kfoldCV_fit_model(data, X_cols, y_col, k = 5 ):
    """
    Perform k-fold cross-validation to fit a model and calculate RMSE.
    
    :param data: DataFrame containing the data
    :param X_cols: List of feature columns
    :param y_col: Target column
    :param k: Number of folds (default is 5)
    :return: RMSE for each fold
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_list = []

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[X_cols].iloc[train_index], data[X_cols].iloc[test_index]
        y_train, y_test = data[y_col].iloc[train_index], data[y_col].iloc[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)

    return rmse_list

def multiple_components(data):
    #for each expNum, print header of the dataframe, and the data frame itself.
    for param in qMRI_params:
        for expNum in data['ExpNum'].unique():
            # only lipid
            x_cols = ['Lipid (fraction)']
            rMSE_lipid =  kfoldCV_fit_model(data[data['ExpNum'] == expNum], x_cols, param)
            # only iron
            x_cols = ['[Fe] (mg/ml)']
            rMSE_iron =  kfoldCV_fit_model(data[data['ExpNum'] == expNum], x_cols, param)
            # lipid,iron
            x_cols = ['Lipid (fraction)', '[Fe] (mg/ml)']
            rMSE_lipid_iron =  kfoldCV_fit_model(data[data['ExpNum'] == expNum],x_cols, param)
            # lipid and lipid*iron
            # first create a new column in the data frame, that is the product of the lipid and iron columns.
            data['Lipid*Iron'] = data['Lipid (fraction)'] * data['[Fe] (mg/ml)']
            x_cols = ['Lipid (fraction)', '[Fe] (mg/ml)', 'Lipid*Iron']
            rMSE_lipid_iron_interaction =  kfoldCV_fit_model(data[data['ExpNum'] == expNum],x_cols, param)
            # boxplot the RMSEs
            plt.figure(figsize=(10, 6))
            plt.title(f'RMSE for expNum {expNum}, {param}')
            plt.boxplot([rMSE_lipid,rMSE_iron, rMSE_lipid_iron, rMSE_lipid_iron_interaction], tick_labels=['Lipid',"iron", 'Lipid+Iron', 'Lipid*Iron'])
            plt.ylabel('RMSE')
            plt.xlabel('Model')
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    # # Read the data file into a pandas dataframe
    data = pd.read_excel('data.xlsx', sheet_name=0)
    # Remove Oshrat experiments
    data = data[~data['ExpNum'].astype(str).str.contains('[a-zA-Z]')]
    # Check pure components 
    # pure_components(data)

    # Check multiple components
    multiple_components(data)

    # Check conbinations of components











