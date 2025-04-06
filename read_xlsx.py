import numpy as np
import matplotlib.pyplot as plt

# ...existing code...

def plot_R1_to_MTV(data, R1, MTV, data_2=None):
    # Perform linear regression on data_1
    x = data[MTV]
    y = data[R1]
    coefficients = np.polyfit(x, y, 1)  # Fit a line (degree=1)
    linear_fit = np.poly1d(coefficients)

    # Plot the regression line
    plt.plot(x, linear_fit(x), color='red', label='Linear fit')
    plt.scatter(data[MTV], data[R1], alpha=0.5, label='Data points')
    plt.xlabel(MTV)
    plt.ylabel(R1)
    plt.title(f'{R1} vs. {MTV}')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Calculate RMSE for the fitted line on data_1 to 
    y_predicted = linear_fit(x)
    rmse_fitted = np.sqrt(np.mean((y - y_predicted) ** 2))
    print(f"RMSE for the fitted line (data_1): {rmse_fitted}")

    # Calculate RMSE of the fitted line from data_1 to the points in data_2
    if data_2 is not None:
        y_predicted_data_2 = linear_fit(data_2[MTV])  # Predict R1 for data_2 using the fitted line
        y_actual_data_2 = data_2[R1]
        rmse_data_2 = np.sqrt(np.mean((y_actual_data_2 - y_predicted_data_2) ** 2))
        print(f"RMSE of the fitted line (data_1) to data_2 points: {rmse_data_2}")
        return rmse_fitted, rmse_data_2

    return rmse_fitted

# Example usage:
# rmse_fitted, rmse_data_2 = plot_R1_to_MTV(data_1, 'R1 (1/sec)', 'MTV (fraction)', data_2=data_2)
# ...existing code...
