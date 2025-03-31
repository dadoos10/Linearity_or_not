def handle_duplicates_and_sort(data_1, data_2, lipid=True):
    """
    This function handles duplicates and sorts the data by Lipid fraction values.
    :return: Filtered and sorted data_1 and data_2
    """
    # Print the first 6 columns of data_1 and data_2
    print(f"data_1: \n{data_1.iloc[:, :6]}.\n data_2: \n{data_2.iloc[:, :6]}")

    # Define the column to use based on the lipid flag
    subset = {True: 'Lipid (fraction)', False: '[Fe] (mg/ml)'}
    column = subset[lipid]

    # Ensure the column data types match and strip whitespace
    data_1[column] = data_1[column].astype(str).str.strip()
    data_2[column] = data_2[column].astype(str).str.strip()

    # Drop duplicates based on the selected column and keep the first occurrence
    data_1 = data_1.drop_duplicates(subset=column, keep='first')
    data_2 = data_2.drop_duplicates(subset=column, keep='first')

    # Find common values in the selected column
    common_values = pd.merge(data_1[[column]], data_2[[column]], on=column, how='inner')[column]

    # Filter rows in both datasets based on common values
    data_1 = data_1[data_1[column].isin(common_values)]
    data_2 = data_2[data_2[column].isin(common_values)]

    # Print the filtered datasets
    print(f"Filtered data_1: \n{data_1}.\nFiltered data_2: \n{data_2}")

    # Sort both datasets by the selected column
    data_1 = data_1.sort_values(by=column)
    data_2 = data_2.sort_values(by=column)

    return data_1, data_2
