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
from collections import defaultdict
from sklearn.metrics import root_mean_squared_error
from scipy.io import savemat
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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




def group_and_mean(data_1, data_2, column_name,param):
    # Group by ExpNum and column_name, calculating the mean for only the qMRI_params columns
    data_1_qMRI = data_1.groupby(["ExpNum", column_name], as_index=False)[[param]].mean(numeric_only=True)
    data_2_qMRI = data_2.groupby(["ExpNum", column_name], as_index=False)[[param]].mean(numeric_only=True)

    # Keep non-numeric columns (if any), retaining the first value for each group (or you can use .last() instead)
    non_qMRI_columns = [col for col in data_1.columns if col != param]

    # Group by ExpNum and column_name, and take the first value for non-numeric columns
    data_1_non_qMRI = data_1.groupby(["ExpNum", column_name], as_index=False)[non_qMRI_columns].first()
    data_2_non_qMRI = data_2.groupby(["ExpNum", column_name], as_index=False)[non_qMRI_columns].first()

    # Merge the mean qMRI_params with the non-numeric columns
    data_1_final = pd.merge(data_1_qMRI, data_1_non_qMRI, on=["ExpNum", column_name])
    data_2_final = pd.merge(data_2_qMRI, data_2_non_qMRI, on=["ExpNum", column_name])
    return data_1_final, data_2_final

def handle_duplicates_and_sort(data_1, data_2, param,lipid=True):
    """
    Handles duplicates and sorts data by Lipid fraction or Fe values.
    
    :param data_1: First DataFrame
    :param data_2: Second DataFrame
    :param lipid: Boolean flag to choose column (True for Lipid fraction, False for Fe)
    :return: Processed data_1 and data_2
    """
    subset = {True: ['Lipid (fraction)'], False: ['[Fe] (mg/ml)']}
    column_name = subset[lipid][0]  # Extract the correct column name
    

    data_1, data_2 = group_and_mean(data_1, data_2, column_name,param)

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
    # return root_mean_squared_error(data[R1].values, data_2[R1].values) # rRMSE!!
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
    png_path = os.path.join(plot_dir, filename)
    plt.savefig(png_path)
    save_as_matlab_format(plot_dir, filename)
    plt.close()


def save_as_matlab_format(plot_dir, filename):
    """
    Save current figure as .svg and raw .mat (x/y data for MATLAB).
    """
    fig = plt.gcf()
    base_name = os.path.splitext(filename)[0]
    fig_path = os.path.join(plot_dir, base_name)

    fig.savefig(fig_path + ".pdf", bbox_inches='tight')

    # Collect raw data from scatter and lines
    xs, ys = [], []
    ax = fig.axes[0]
    for pc in ax.collections:  # scatterplot points
        offsets = pc.get_offsets()
        if offsets.size:
            xs.extend(offsets[:, 0])
            ys.extend(offsets[:, 1])
    for line in ax.lines:      # line plot data
        xs.extend(line.get_xdata())
        ys.extend(line.get_ydata())

    # Save as .mat if data was found
    if xs:
        savemat(fig_path + ".mat", {"x": xs, "y": ys})


def run_scan_rescan(data,exps_pair,rmse_dict,lipid = True):
    rmse_table = pd.DataFrame(
        data={col: [[] for _ in range(len(qMRI_params))] for col in ["Fitted RMSE", "Scan-Rescan RMSE"]},
        index=qMRI_params
    )
    for param in qMRI_params:
        ax = None  # Initialize ax to None for each parameter
        data = data_dict[param]  # Get the DataFrame for the current MRI parameter
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

            data_1,data_2 = handle_duplicates_and_sort(data_1, data_2,param,lipid)
            if data_1.empty or data_2.empty:
                print("⚠️ plot_qMRI_to_bio skipped: one of the inputs is empty.")
                continue
            fig,ax,rmse_fitted, scan_rescan_rmse, (param_name, tissue_col, tissue_type) =  plot_qMRI_to_bio(data_1, data_2, param, exps_pair,i, lipid = lipid)
            rmses.append(rmse_fitted)
            # if isinstance(rmse_table.loc[param, "Fitted RMSE"], list):
            #     rmse_table.loc[param, "Fitted RMSE"].append(rmse_fitted)
            # else:
            #     rmse_table.loc[param, "Fitted RMSE"] = [rmse_fitted]

        # mean_rmse_fitted = np.mean(rmse_table.loc[param, "Fitted RMSE"]) if rmse_table.loc[param, "Fitted RMSE"] else 0
        mean_rmse_fitted = np.mean(rmses) if rmses else 0
        if ax is None:
            print("⚠️ No valid data to plot.")
            continue
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
        # plt.show()
        plt.close()

def pure_components(data):
    exp_lipid_all_to_check  = [PC_all,PC_Cholest_all,PC_SM_all]
    exp_iron_all_to_check = [Fe2_all, Fe3_all, Ferittin_all, Tranferrin_all]
    rmse_dict = {}
    for iron_pair in exp_iron_all_to_check:
        rmse_dict = run_scan_rescan(data,iron_pair, rmse_dict,lipid = False)
    
    dict_to_boxplot(rmse_dict)
    rmse_dict = {}
   
    for lipid_pair in exp_lipid_all_to_check:
        rmse_dict = run_scan_rescan(data,lipid_pair,rmse_dict)

    dict_to_boxplot(rmse_dict)

def kfoldCV_fit_model(data, X_cols, y_col, k = 4 ):
    """
    Perform k-fold cross-validation to fit a model and calculate RMSE.
    
    :param data: DataFrame containing the data
    :param X_cols: List of feature columns
    :param y_col: Target column
    :param k: Number of folds (default is 5)
    :return: RMSE for each fold
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    # initialize an empty list with wheits of the model
    weights = np.zeros(len(X_cols))
    rmse_list = []

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[X_cols].iloc[train_index], data[X_cols].iloc[test_index]
        y_train, y_test = data[y_col].iloc[train_index], data[y_col].iloc[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)
        # Get the weights of the model
        weights += model.coef_
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)
    weights /= k  # Average the weights over k folds
    rounded_weights = np.round(weights, 2)
    return rmse_list, rounded_weights

def plot_the_boxes(
    iron_type, lipid_type, expNum, param,
    rMSE_lipid, rMSE_iron, rMSE_lipid_iron, rMSE_lipid_iron_interaction,
    weights_lipid, weights_iron, weights_lipid_iron, weights_lipid_iron_interaction,
    non_zero
):
    from statannotations.Annotator import Annotator
    from scipy.stats import kruskal

    # Prepare data for boxplot and annotation
    all_rmse = rMSE_lipid + rMSE_iron + rMSE_lipid_iron + rMSE_lipid_iron_interaction
    model_labels = (
        ["Lipid"] * len(rMSE_lipid) +
        ["Iron"] * len(rMSE_iron) +
        ["Lipid+Iron"] * len(rMSE_lipid_iron) +
        ["Lipid*Iron"] * len(rMSE_lipid_iron_interaction)
    )
    df = pd.DataFrame({
        "RMSE": all_rmse,
        "Model": model_labels
    })

    # Create figure
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="Model", y="RMSE", data=df)

    # Define pairwise comparisons
    pairs = [
        ("Lipid", "Iron"),
        ("Iron", "Lipid+Iron"),
        ("Lipid+Iron", "Lipid*Iron"),
        ("Lipid", "Lipid*Iron")
    ]

    # Add statistical annotations
    annotator = Annotator(ax, pairs, data=df, x="Model", y="RMSE")
    annotator.configure(test='Mann-Whitney', 
                        text_format='star', loc='outside',
                        line_height = 0.01,
                        verbose=0,hide_non_significant=True) 
    annotator.apply_and_annotate()

    # Adjust y-limits for annotation visibility (smarter buffer)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_buffer = min(0.15 * y_range, 0.1 * y_max)  # Cap excessive padding
    ax.set_ylim(y_min, y_max + 0.2*(y_max - y_min))

    # Clean axis labels
    tick_labels = [
        f"Lipid\n[{', '.join(f'{w:.2f}' for w in weights_lipid)}]",
        f"Iron\n[{', '.join(f'{w:.2f}' for w in weights_iron)}]",
        f"Lipid+Iron\n[{', '.join(f'{w:.2f}' for w in weights_lipid_iron)}]",
        f"Lipid*Iron\n[{', '.join(f'{w:.2f}' for w in weights_lipid_iron_interaction)}]"
    ]
    ax.set_xticklabels(tick_labels)

    plt.title(f'RMSE for expNum {expNum}, {param}\niron type: {iron_type}, lipid type: {lipid_type}', fontsize = 11)
    plt.ylabel('RMSE')
    plt.xlabel('Model')
    plt.grid(True)
    plt.subplots_adjust(top=0.82)
      # Add space at top

    # Save the plot
    if non_zero:
        plot_dir = f'plots/multiple_components/non-zero/exp{expNum}'
    else:
        plot_dir = f'plots/multiple_components/exp{expNum}'

    filename = f"{param}_{expNum}.png".replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
    save_file(plot_dir, filename)


def multiple_components(data, non_zero = True):
    rmse_dict = defaultdict(list)  # key: (lipid_type, iron_type), value: list of 4 median RMSEs

    for param in qMRI_params:
        data = data_dict[param]  # Get the DataFrame for the current MRI parameter
        if non_zero:
        # Filter out rows where [Fe] (mg/ml) or Lipid (fraction) is zero
            data = data[
            (data["[Fe] (mg/ml)"] != 0) &
            (data["Lipid (fraction)"] != 0)
            ]
        #for each expNum, print header of the dataframe, and the data frame itself.
        data['Lipid*Iron'] = data['Lipid (fraction)'] * data['[Fe] (mg/ml)']
        for expNum in data['ExpNum'].unique():
            cur_data = data[data['ExpNum'] == expNum]
            if cur_data.shape[0] < 7:
                print(f"⚠️ Not enough data for expNum {expNum} in {param}. Skipping...")
                continue
            # only lipid
            x_cols = ['Lipid (fraction)']
            rMSE_lipid,weights_lipid =  kfoldCV_fit_model(cur_data, x_cols, param)
            # only iron
            x_cols = ['[Fe] (mg/ml)']
            rMSE_iron,weights_iron =  kfoldCV_fit_model(cur_data, x_cols, param)
            # lipid,iron
            x_cols = ['Lipid (fraction)', '[Fe] (mg/ml)']
            rMSE_lipid_iron,weights_lipid_iron =  kfoldCV_fit_model(cur_data,x_cols, param)
            # lipid and lipid*iron
            # first create a new column in the data frame, that is the product of the lipid and iron columns.
            # data['Lipid*Iron'] = data['Lipid (fraction)'] * data['[Fe] (mg/ml)']
            x_cols = ['Lipid (fraction)', '[Fe] (mg/ml)', 'Lipid*Iron']
            rMSE_lipid_iron_interaction,weights_lipid_iron_interaction =  kfoldCV_fit_model(cur_data,x_cols, param)
            iron_type = cur_data["Iron type"].iloc[0]
            lipid_type = cur_data["Lipid type"].iloc[0]

            plot_the_boxes(iron_type,lipid_type, expNum, param, rMSE_lipid, rMSE_iron, rMSE_lipid_iron, 
                           rMSE_lipid_iron_interaction, weights_lipid, weights_iron, weights_lipid_iron, weights_lipid_iron_interaction,non_zero)
           
            # Save median RMSEs for summary
            medians = [np.median(rMSE_lipid), np.median(rMSE_iron), np.median(rMSE_lipid_iron), np.median(rMSE_lipid_iron_interaction)]
            rmse_dict[(lipid_type, iron_type)].append(medians)
        plot_summary_rmse(rmse_dict,param,non_zero)

def plot_summary_rmse(rmse_dict,param,non_zero):
    from collections import defaultdict

    # Group by lipid type
    grouped_data = defaultdict(list)  # lipid_type → list of (iron_type, avg_RMSEs)
    for (lipid_type, iron_type), rmse_lists in rmse_dict.items():
        # # filter a littlebit
        # if iron_type in ["Ferr+Trans", "Ferritin"]:
        #     continue  # Skip these iron types
        avg_rmse = np.mean(rmse_lists, axis=0)
        grouped_data[lipid_type].append((iron_type, avg_rmse))

    # Marker styles for each model
    model_labels = ['Lipid only', 'Iron only', 'Lipid+Iron', 'Lipid+Iron+Interaction']
    markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
    colors = ['black', 'blue', 'green', 'red']

    if non_zero:
        summary_dir = f'plots/summary/non-zero/{param}/'
    else:
        summary_dir = f'plots/summary/{param}/'
    for lipid_type, entries in grouped_data.items():
        iron_types = [e[0] for e in entries]
        rmse_values = np.array([e[1] for e in entries])  # shape: (num_iron_types, 4)
        x = np.arange(len(iron_types))

        plt.figure(figsize=(12, 6))

        for i in range(4):  # for each model type
            plt.scatter(
                x + i * 0.1 - 0.15,  # small horizontal shift for separation
                rmse_values[:, i],
                marker=markers[i],
                color=colors[i],
                label=model_labels[i],
                s=80  # marker size
            )

        plt.xticks(x, iron_types, rotation=45)
        plt.ylabel('Median RMSE')
        plt.title(f'Summary RMSE for Lipid Type: {lipid_type}, {param}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        filename = f'summary_RMSE_non-zeros_{lipid_type}.png'.replace(" ", "_")
        save_file(summary_dir, filename)

def sanitize_filename(filename):
    """
    Remove or replace characters that are invalid in filenames (especially on Windows).
    """
    import re
    return re.sub(r'[<>:"/\\|?*\[\]\(\)\s]+', '_', filename)

def plot_with_regression(data, x_col, y_col, group_col, title, xlabel, ylabel, plot_dir, filename_prefix, return_stats=False):
    """
    Plot y_col vs. x_col for each group in group_col, fit linear regression, and save the figure.

    Parameters:
    - data: DataFrame containing the data
    - x_col, y_col: columns for X and Y axes
    - group_col: column by which to group and color data (e.g., 'Lipid type', 'Iron type')
    - title, xlabel, ylabel: plot labels
    - plot_dir: directory to save the plot
    - filename_prefix: filename base prefix for saving
    """
    stats = {}
    plt.figure(figsize=(10, 6))
    for group in data[group_col].dropna().unique():
        subset = data[data[group_col] == group]
        if subset.empty:
            continue
        x = subset[x_col].values.reshape(-1, 1)
        y = subset[y_col].values
        model = LinearRegression().fit(x, y)
        y_pred = model.predict(x)
        r2 = r2_score(y, y_pred)
        slope = model.coef_[0]
        intercept = model.intercept_
        print(f"{y_col} | {group_col}: {group} | Slope: {slope:.4f}, Intercept: {intercept:.4f}, R²: {r2:.4f}")
        sns.scatterplot(x=x.flatten(), y=y, label=f"{group}", alpha=0.5)
        plt.plot(x.flatten(), y_pred, linestyle='--')
        if return_stats:
            stats[group] = (intercept, slope, r2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(title=group_col)
    filename = f"{sanitize_filename(filename_prefix)}-vs-{sanitize_filename(x_col)}.png"
    save_file(plot_dir, filename)
    if return_stats:
        return stats

def preanalysis(data_dict):
    all_stats = []
    for param, data in data_dict.items():
                # Plot vs. Lipid (fraction) when iron concentrations are zero
        lipid_data = data[(data[Fe_protein_con] == 0) & (data[Fe_con] == 0)]
        lipid_stats = plot_with_regression(
            data=lipid_data,
            x_col='Lipid (fraction)',
            y_col=param,
            group_col='Lipid type',
            title=f'{param} vs. Lipid Fraction n = {len(lipid_data)}',
            xlabel='Lipid Fraction',
            ylabel=param,
            plot_dir='plots/preanalysis/lipid_fraction',
            filename_prefix=param[:3],
            return_stats=True
        )
        for group, (intercept, slope, r2) in lipid_stats.items():
            all_stats.append({
                "group_type": "Lipid type",
                "group": group,
                "MRI_param": param,
                "intercept": intercept,
                "slope": slope,
                "r2": r2
            })

        # Plot vs. Iron when Lipid = 0
        iron_data_no_lipid = data[data[Lipid_con] == 0]
        iron_stats_0 = plot_with_regression(
            data=iron_data_no_lipid,
            x_col='[Fe] (mg/ml)',
            y_col=param,
            group_col='Iron type',
            title=f'{param} vs. [Fe] (mg/ml) n = {len(iron_data_no_lipid)}',
            xlabel='[Fe] (mg/ml)',
            ylabel=param,
            plot_dir='plots/preanalysis/iron_concentration/no_lipid',
            filename_prefix=param[:3],
            return_stats=True
        )
        for group, (intercept, slope, r2) in iron_stats_0.items():
            all_stats.append({
                "group_type": "Iron type (lipid=0)",
                "group": group,
                "MRI_param": param,
                "intercept": intercept,
                "slope": slope,
                "r2": r2
            })

        # Plot vs. Iron when Lipid = 0.25
        iron_data_lipid25 = data[data[Lipid_con] == 0.25]
        plot_with_regression(
            data=iron_data_lipid25,
            x_col='[Fe] (mg/ml)',
            y_col=param,
            group_col='Iron type',
            title=f'{param} vs. [Fe] (mg/ml), 25% Lipid n = {len(iron_data_lipid25)}',
            xlabel='[Fe] (mg/ml)',
            ylabel=param,
            plot_dir='plots/preanalysis/iron_concentration/25_percent_lipid',
            filename_prefix=param[:3]
        )
    stats_df = pd.DataFrame(all_stats)
        # Pivot the table to match your desired structure
    pivot_df = stats_df.pivot_table(
        index=["group_type", "group"],
        columns="MRI_param",
        values=["intercept", "slope", "r2"]
    )
        # Flatten multi-index columns
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df.reset_index(inplace=True)
    pivot_df = pivot_df.round(2)

    return pivot_df

def data_preprcess(data):
    data = data[~data['ExpNum'].astype(str).str.contains('[a-zA-Z]')]
    data = data[~((data['ExpNum'] == 6) | (data['ExpNum'] == 11) |(data['ExpNum'] == 13) )] # 11 was neglected due to lipid concetration mistake, 6 was neglected lipid type might be wrong
    return data

def mixed_linear_model(data, X_cols, y_col):
    X = data[X_cols]
    y = data[y_col]

    import statsmodels.api as sm

    # Add constant (intercept)
    X_with_intercept = sm.add_constant(X)

    # Fit OLS model
    ols_model = sm.OLS(y, X_with_intercept).fit()

    # Get results
    r_squared = ols_model.rsquared
    p_values = ols_model.pvalues
    coefficients = ols_model.params  # includes intercept
    overall_p_value = ols_model.f_pvalue         # p-value for the overall model (F-test)


    # Optional: full summary
    print(ols_model.summary())
    return r_squared, coefficients, p_values, overall_p_value

def multiple_components_mixed(data, non_zero = True):
    if non_zero:
        # Filter out rows where [Fe] (mg/ml) or Lipid (fraction) is zero
        data = data[
        (data["[Fe] (mg/ml)"] != 0) &
        (data["Lipid (fraction)"] != 0)
        ]
    #for each expNum, print header of the dataframe, and the data frame itself.
    data['Lipid*Iron'] = data['Lipid (fraction)'] * data['[Fe] (mg/ml)']
    rmse_dict = defaultdict(list)  # key: (lipid_type, iron_type), value: list of 4 median RMSEs

    for param in qMRI_params:
        for expNum in data['ExpNum'].unique():
            cur_data = data[data['ExpNum'] == expNum]
            # only lipid
            x_cols = ['Lipid (fraction)']
            r_squared, coefficients, p_values, overall_p_value =  mixed_linear_model(cur_data, x_cols, param)
            # only iron
            x_cols = ['[Fe] (mg/ml)']
            r_squared, coefficients, p_values, overall_p_value =  mixed_linear_model(cur_data, x_cols, param)
            # lipid,iron
            x_cols = ['Lipid (fraction)', '[Fe] (mg/ml)']
            r_squared, coefficients, p_values, overall_p_value =  mixed_linear_model(cur_data,x_cols, param)
            # lipid and lipid*iron
            x_cols = ['Lipid (fraction)', '[Fe] (mg/ml)', 'Lipid*Iron']
            r_squared, coefficients, p_values, overall_p_value =  mixed_linear_model(cur_data,x_cols, param)
            iron_type = cur_data["Iron type"].iloc[0]
            lipid_type = cur_data["Lipid type"].iloc[0]

            # plot_the_boxes(iron_type,lipid_type, expNum, param, rMSE_lipid, rMSE_iron, rMSE_lipid_iron, 
            #                rMSE_lipid_iron_interaction, weights_lipid, weights_iron, weights_lipid_iron, weights_lipid_iron_interaction,non_zero)
           
            # Save median RMSEs for summary
            # medians = [np.median(rMSE_lipid), np.median(rMSE_iron), np.median(rMSE_lipid_iron), np.median(rMSE_lipid_iron_interaction)]
            rmse_dict[(lipid_type, iron_type)].append(medians)
        plot_summary_rmse(rmse_dict,param,non_zero)
def round_up_smart(x):
    """
    Round up to the nearest meaningful decimal step based on magnitude.
    Examples:
        0.762 -> 0.8
        0.044 -> 0.05
        0.003 -> 0.01
    """
    print(f"rounding up {x}")
    order = np.floor(np.log10(x))
    step = 10 ** (order - 1)
    rounded_value = np.ceil(x / step) * step
    print(f"rounded value: {rounded_value}")
    return rounded_value if rounded_value > 0 else step  # Ensure we don't return zero

def crop_df(data, y_col, crop = True):
    if not crop:
        # If cropping is not needed, return the original data and an empty dictionary
        return data.copy(), {}
    max_effects = {}
    keep_indices = set(data.index)

    for expNum in data['ExpNum'].unique():
        exp_data = data[data['ExpNum'] == expNum]
        # lipid_effect = exp_data[exp_data['[Fe] (mg/ml)'] == 0][y_col]

        # if not lipid_effect.empty:
        #     max_effect = round_up_smart(lipid_effect.max())*2
        #     max_effects[expNum] = max_effect

        #     drop_idx = data[(data['ExpNum'] == expNum) & (data[y_col] > max_effect)].index
        #     keep_indices -= set(drop_idx)
                # Decide which column to base cropping on
        iron_type = str(exp_data['Iron type'].iloc[0]).lower()
        if 'ferritin' in iron_type or 'transferrin' in iron_type:
            col_to_crop = '[Protein](mg/ml)'
        else:
            col_to_crop = '[Fe] (mg/ml)'

        if col_to_crop not in exp_data.columns:
            continue  # skip if column missing

        unique_vals = sorted(exp_data[col_to_crop].dropna().unique(), reverse=True)        
        # Store the top 2 Fe levels (if available)
        top_to_remove = unique_vals[:2]  # max and second max
        max_effects[expNum] = top_to_remove

        # Drop all rows with Fe in top 2 values
        drop_idx = exp_data[exp_data[col_to_crop].isin(top_to_remove)].index
        keep_indices -= set(drop_idx)

    # Return only the rows that passed the filter, keeping their original index
    data_cropped = data.loc[sorted(keep_indices)].copy()
    return data_cropped, max_effects

def crop(data):
    # Define shared columns and target variables
    shared_cols = ['ExpNum', "Iron type", 'Lipid type', 'Lipid (fraction)', '[Protein](mg/ml)', '[Fe] (mg/ml)']
    crop = True  # Set to False if you want to skip cropping
    # Apply cropping to each signal
    data_R1, max_R1 = crop_df(data[shared_cols + [R1]].copy(), R1,crop)
    data_R2, max_R2 = crop_df(data[shared_cols + [R2]].copy(), R2,crop)
    data_R2s, max_R2s = crop_df(data[shared_cols + [R2s]].copy(), R2s,crop)
    data_MT, max_MT = crop_df(data[shared_cols + [MT]].copy(), MT,crop)

    # Find common indices across all four cropped dataframes
    common_idx = set(data_R1.index) & set(data_R2.index) & set(data_R2s.index) & set(data_MT.index)

    # Build summary table
    all_exp = sorted(set(max_R1) | set(max_R2) | set(max_R2s) | set(max_MT))
    summary_rows = []

    
    for exp in all_exp:
        indices = [
            idx for idx in common_idx if data_R1.loc[idx, 'ExpNum'] == exp
        ]

        def count_valid_rows(df):
            df_exp = df[df['ExpNum'] == exp]
            return df_exp[
                (df_exp['Lipid (fraction)'] != 0) &
                ((df_exp['[Fe] (mg/ml)'] != 0) | (df_exp['[Protein](mg/ml)'] != 0))
            ].shape[0]
        def count_rows(df):
            # Count rows for a specific experiment
            df_exp = df[df['ExpNum'] == exp]
            return df_exp.shape[0]

        summary_rows.append({
            'ExpNum': exp,
            'Max R1': max_R1.get(exp, np.nan),
            'Max R2': max_R2.get(exp, np.nan),
            'Max R2s': max_R2s.get(exp, np.nan),
            'Max MT': max_MT.get(exp, np.nan),
            'Common Indices': indices,
            'R1 Rows (non-zero)': f"{count_rows(data_R1)}({count_valid_rows(data_R1)})",
            'R2 Rows (non-zro)': f"{count_rows(data_R2)}({count_valid_rows(data_R2)})",
            'R2s Rows (non-zero)': f"{count_rows(data_R2s)}({count_valid_rows(data_R2s)})",
            'MT Rows (non-zero)': f"{count_rows(data_MT)}({count_valid_rows(data_MT)})"
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save everything
    with pd.ExcelWriter('data_cropped.xlsx') as writer:
        data_R1.to_excel(writer, sheet_name='R1', index=False)
        data_R2.to_excel(writer, sheet_name='R2', index=False)
        data_R2s.to_excel(writer, sheet_name='R2s', index=False)
        data_MT.to_excel(writer, sheet_name='MT', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    return data_R1, data_R2, data_R2s, data_MT, summary_df



if __name__ == "__main__":
    ### Load data ###
    print("Loading data from 'data.xlsx'...")
    data = pd.read_excel('data.xlsx', sheet_name=0)
    data = data_preprcess(data)
    data_R1, data_R2, data_R2s, data_MT, summary_df = crop(data)
    print("✅ Data loaded successfully")
    
    #### preanalysis #####
    print("Running preanalysis...")
    data_dict = {
        R1: data_R1,
        R2: data_R2,
        R2s: data_R2s,
        MT: data_MT
    }
    preanalysis_table = preanalysis(data_dict)
    # preanalysis_table.to_excel("plots/preanalysis/regression_summary_table.xlsx", index=False)
    # print("✅ Summary table saved to 'regression_summary_table.xlsx'")

    ##### pure components, also called AIM 1 #####
    print("Running pure components analysis...")
    # pure_components(data_dict)

    ##### multiple components, also called AIM 2 #####
    print("Running multiple components analysis...")
    rmse_dict = multiple_components(data_dict, non_zero=True)
    
    # Another way to analyse is as mixed linear model. we check R_squered and p-value of model without k-fold cross validation.
    print("Running mixed linear model analysis...")
    # multiple_components_mixed(data, non_zero=False)

    
    print("done")