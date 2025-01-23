import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot

# Load the data
file_path = "Nat_Gas.csv"  # Replace with the actual file path
data = pd.read_csv(file_path)

# Ensure the 'Dates' column is in datetime format
data['Dates'] = pd.to_datetime(data['Dates'])

# Sort the data by date (if not already sorted)
data = data.sort_values(by="Dates")

# Set 'Dates' as the index
data.set_index("Dates", inplace=True)

# Visualize the historical data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["Prices"], marker="o", label="Historical Prices")
plt.title("Natural Gas Prices (Historical Data)", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
# plt.show()
plt.savefig("Nat_Gas_historical.png")

# Plot autocorrelation to analyze lags
autocorrelation_plot(data["Prices"])
plt.title("Autocorrelation of Prices", fontsize=16)
plt.tight_layout()
# plt.show()
plt.savefig("Autocorrelation_of_Prices.png")

# Fit an ARIMA model
p, d, q = 2, 1, 2  # Adjust these parameters based on your data
model = ARIMA(data["Prices"], order=(p, d, q))
arima_result = model.fit()

# Print model summary
print(arima_result.summary())

# Forecast for the next 12 months
forecast_steps = 12
forecast = arima_result.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq="M")[1:]
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Combine historical and forecasted data for visualization
forecast_data = pd.DataFrame({
    "Forecast": forecast_mean.values,
    "Lower Bound": forecast_conf_int.iloc[:, 0].values,
    "Upper Bound": forecast_conf_int.iloc[:, 1].values
}, index=forecast_index)

# Visualize historical and forecasted data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["Prices"], marker="o", label="Historical Prices")
plt.plot(forecast_data.index, forecast_data["Forecast"], marker="x", linestyle="--", label="Forecasted Prices")
plt.fill_between(forecast_data.index, forecast_data["Lower Bound"], forecast_data["Upper Bound"], color="gray", alpha=0.3, label="Confidence Interval")
plt.title("Natural Gas Prices (Historical and Forecasted)", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
# plt.show()
plt.savefig("Forecasted_prices.png")

# Function to estimate price for a given date
def estimate_price_arima(input_date):
    input_date = pd.to_datetime(input_date)
    if input_date in forecast_data.index:
        return forecast_data.loc[input_date, "Forecast"]
    else:
        print("Date is out of forecast range. Please provide a date within the forecasted period.")
        return None

# Example: Estimate price for a specific date
input_date = "2025-06-30"  # Replace with the desired date
estimated_price = estimate_price_arima(input_date)
if estimated_price is not None:
    print(f"Estimated price for {input_date}: ${estimated_price:.2f}")




def price_storage_contract(
    injection_dates, withdrawal_dates, 
    prices, injection_rate, withdrawal_rate, 
    max_volume, storage_cost_per_unit
):
    """
    Calculate the value of a gas storage contract.

    Parameters:
    - injection_dates (list of str): Dates for injecting gas.
    - withdrawal_dates (list of str): Dates for withdrawing gas.
    - prices (dict): Dictionary mapping dates (str) to prices (float).
    - injection_rate (float): Rate of gas injection (units per day).
    - withdrawal_rate (float): Rate of gas withdrawal (units per day).
    - max_volume (float): Maximum volume of gas that can be stored.
    - storage_cost_per_unit (float): Cost of storing gas (per unit).

    Returns:
    - float: Net value of the storage contract.
    """
    # Convert dates to pandas datetime for consistency
    injection_dates = pd.to_datetime(injection_dates)
    withdrawal_dates = pd.to_datetime(withdrawal_dates)
    
    # Initialize variables
    current_volume = 0
    total_injection_cost = 0
    total_withdrawal_revenue = 0
    total_storage_cost = 0
    
    # Process injection
    for date in injection_dates:
        if date.strftime('%Y-%m-%d') not in prices:
            raise ValueError(f"Price data not available for injection date {date}.")
        
        injection_price = prices[date.strftime('%Y-%m-%d')]
        injection_volume = min(injection_rate, max_volume - current_volume)
        
        current_volume += injection_volume
        total_injection_cost += injection_volume * injection_price
        total_storage_cost += injection_volume * storage_cost_per_unit
    
    # Process withdrawal
    for date in withdrawal_dates:
        if date.strftime('%Y-%m-%d') not in prices:
            raise ValueError(f"Price data not available for withdrawal date {date}.")
        
        withdrawal_price = prices[date.strftime('%Y-%m-%d')]
        withdrawal_volume = min(withdrawal_rate, current_volume)
        
        current_volume -= withdrawal_volume
        total_withdrawal_revenue += withdrawal_volume * withdrawal_price
    
    # Calculate net value
    net_value = total_withdrawal_revenue - total_injection_cost - total_storage_cost
    
    return net_value

# Example usage
prices = {
    "2024-01-31": 50.0,
    "2024-02-28": 52.0,
    "2024-03-31": 55.0,
    "2024-04-30": 53.0,
    "2024-05-31": 54.0,
    "2024-06-30": 58.0,
    "2024-07-31": 60.0,
    "2024-08-31": 62.0,
    "2024-09-30": 65.0,
}

# Inputs
injection_dates = ["2024-01-31", "2024-02-28", "2024-03-31"]
withdrawal_dates = ["2024-07-31", "2024-08-31", "2024-09-30"]
injection_rate = 1000  # units per day
withdrawal_rate = 1000  # units per day
max_volume = 3000  # units
storage_cost_per_unit = 0.5  # cost per unit

# Calculate contract value
contract_value = price_storage_contract(
    injection_dates, withdrawal_dates, prices, 
    injection_rate, withdrawal_rate, max_volume, storage_cost_per_unit
)

print(f"Net value of the storage contract: ${contract_value:.2f}")
