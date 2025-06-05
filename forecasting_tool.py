import numpy as np
import pandas as pd
import requests
from io import StringIO
from datetime import datetime

import yfinance as yf

# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

# Data preprocessing and evaluation
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Deep learning models
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Financial indicators
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Anomaly detection
from scipy.stats import zscore

import pandas_datareader as pdr
from pandas_datareader import data as web
import datetime as dt

from skforecast.recursive._forecaster_recursive import ForecasterRecursive


leading_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
tickers = {
    'target_ticker': 'AAPL',        # המניה שברצונך לחזות
    'Nasdaq_100': 'QQQ',            # מדד מניות טכנולוגיה מובילות – הכי רלוונטי ל-AAPL
    'Finance_Sector': 'XLF',        # נותן הקשר לסקטור הפיננסי והמצב הכלכלי
    'Europe_Stocks': 'FEZ',         # חשיפה לאירופה (אפשר גם VGK)
    'Gold_ETF': 'GLD',              # אינדיקציה להעדפת סיכון / גידור אינפלציה
}

def download_data(ticker='AAPL', start='2020-01-01', end='2025-04-01'):
	"""
	Download stock data from Yahoo Finance for a single ticker.
	
	Args:
		ticker (str): The stock ticker symbol
		start (str): Start date for data collection
		end (str): End date for data collection
	
	Returns:
		DataFrame: DataFrame with date as index and Close price as the target_ticker column
	"""
	data = yf.download(ticker, start=start, end=end)
	stock_data = pd.DataFrame()
	stock_data['target_ticker'] = data['Close']
	
	return stock_data

def get_and_prepare_unemployment_data():
    """
    Download U.S. unemployment rate data (UNRATE) from FRED,
    resample to daily frequency, and interpolate linearly.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception("Failed to download unemployment data from FRED.")

    data = StringIO(response.text)
    # print()
    unemployment_data = pd.read_csv(data, parse_dates=['observation_date'], index_col='observation_date')

    unemployment_data.columns = ['UNRATE']
    unemployment_data['Unemployment Rate'] =  unemployment_data['UNRATE'] # Convert to decimal format

    # unemployment_data = unemployment_data.resample("D").interpolate(method='linear')

    return unemployment_data

tickers_data = download_data(tickers)

employment_data = get_and_prepare_unemployment_data()

# First, add year and month columns to both dataframes
tickers_data['year'] = tickers_data.index.year
tickers_data['month'] = tickers_data.index.month
employment_data['year'] = employment_data.index.year
employment_data['month'] = employment_data.index.month

# The employment data is already monthly, so we can use it directly
# Merge based on year and month
merged_df = pd.merge(
    tickers_data,
    employment_data[['Unemployment Rate', 'year', 'month']],  # Select only needed columns
    on=['year', 'month'],
    how='left'
)

# Preserve the original index from tickers_data
merged_df.index = tickers_data.index

# Drop the temporary columns used for merging
merged_df = merged_df.drop(columns=['year', 'month'])

def add_technical_indicators(df):
    """
    Add RSI and MACD indicators to the DataFrame.
    """
    # Initialize the RSI and MACD indicators
    rsi_indicator = RSIIndicator(df['target_ticker'])
    macd_indicator = MACD(df['target_ticker'])

    # Calculate RSI and MACD
    df['RSI'] = rsi_indicator.rsi()
    df['MACD'] = macd_indicator.macd() # Moving Average Convergence Divergence
    df['MACD Signal'] = macd_indicator.macd_signal()
    df['MACD Histogram'] = macd_indicator.macd_diff()

    return df

final_merged_df = add_technical_indicators(merged_df)

# Count and display missing values in the DataFrame before removing them
missing_values = final_merged_df.isna().sum()
missing_percent = (missing_values / len(final_merged_df)) * 100
missing_stats = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percent
})

# Display results
print("Missing Values Summary:")
print(missing_stats[missing_stats['Missing Values'] > 0])  # Only show columns with missing values
print(f"\nTotal rows in dataset: {len(final_merged_df)}")
print(f"Rows with at least one missing value: {final_merged_df.isna().any(axis=1).sum()}")
print(f"Percentage of rows with missing values: {final_merged_df.isna().any(axis=1).sum() / len(final_merged_df) * 100:.2f}%")

final_merged_df = final_merged_df[~final_merged_df.isna().any(axis=1)]  # Check for NaN values


# Count and display missing values in the DataFrame before removing them
missing_values = final_merged_df.isna().sum()
missing_percent = (missing_values / len(final_merged_df)) * 100
missing_stats = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percent
})

# Display results
print("Missing Values Summary:")
print(missing_stats[missing_stats['Missing Values'] > 0])  # Only show columns with missing values
print(f"\nTotal rows in dataset: {len(final_merged_df)}")
print(f"Rows with at least one missing value: {final_merged_df.isna().any(axis=1).sum()}")
print(f"Percentage of rows with missing values: {final_merged_df.isna().any(axis=1).sum() / len(final_merged_df) * 100:.2f}%")

# Basic statistics: min, max, mean, etc.
summary_stats = final_merged_df.describe()
print("Summary Statistics:\n", summary_stats)

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(final_merged_df)

# Create a DataFrame with the scaled data to preserve column names
scaled_df = pd.DataFrame(scaled_data, columns=final_merged_df.columns, index=final_merged_df.index)

# Enhanced visual detection of outliers using a boxplot with seaborn
plt.figure(figsize=(16, 10))
sns.set_style("whitegrid")  # Set the seaborn style

# Create the boxplot with enhanced styling
ax = sns.boxplot(
    data=scaled_df,
    palette="Set3",        # Use a colorful palette
    width=0.6,             # Adjust box width
    fliersize=3,           # Size of outlier points
    linewidth=1.5          # Width of the box lines
)

# Improve the plot appearance
plt.title('Outlier Detection with Standardized Data', fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylabel('Standardized Value', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a horizontal line at 0
plt.axhline(y=0, color='grey', linestyle='--', alpha=0.3)

# Add a reference for potential outliers
plt.axhline(y=3, color='red', linestyle=':', alpha=0.5, label="3σ threshold")
plt.axhline(y=-3, color='red', linestyle=':', alpha=0.5)

plt.legend()
plt.tight_layout()
plt.show()


from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
# Run Isolation Forest
iso = IsolationForest(contamination=0.01, random_state=42)
outlier_flags = iso.fit_predict(scaled_df)  # -1 = outlier, 1 = inlier

# Add labels to the DataFrame
scaled_df['Outlier'] = outlier_flags

# Project to 2D using PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_df.drop(columns='Outlier'))

pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=scaled_df.index)
pca_df['Outlier'] = scaled_df['Outlier']

plt.figure(figsize=(12, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Outlier', palette={1: 'blue', -1: 'red'}, alpha=0.6)
plt.title("Isolation Forest Outlier Detection (PCA Visualization)")
plt.legend(title='Outlier', loc='upper right')
plt.grid(True)
plt.show()

# Compute correlation between all variables
correlation_matrix = final_merged_df.corr()

# Enhanced visualization of the correlation matrix
plt.figure(figsize=(14, 10))
mask = np.triu(correlation_matrix)  # Create a mask for the upper triangle
sns.heatmap(
    correlation_matrix,
    annot=True,           # Show values in cells
    cmap='coolwarm',      # Color scheme
    fmt='.2f',            # Format numbers to 2 decimal places
    linewidths=0.5,       # Add line width between cells
    mask=mask,            # Apply the mask to show only lower triangle
    vmin=-1, vmax=1       # Fix the scale from -1 to 1
)
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()        # Adjust layout to make room for labels
plt.show()


# Statistical Analysis on Returns
# Let's analyze the daily returns to understand risk characteristics

# Calculate daily returns
returns = final_merged_df['target_ticker'].pct_change()
returns = returns.dropna()

# Basic statistics of returns
mean_return = returns.mean()
std_return = returns.std()

# Calculate probabilities
prob_positive_return = (returns > 0).mean()
prob_negative_return = (returns < 0).mean()

# Plot returns distribution
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
returns.hist(bins=50, alpha=0.7)
plt.axvline(x=mean_return, color='r', linestyle='--', label=f'Mean: {mean_return:.4f}')
plt.title('Distribution of Daily Returns')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(returns, fill=True)
plt.axvline(x=mean_return, color='r', linestyle='--', label=f'Mean: {mean_return:.4f}')
plt.axvline(x=mean_return + 2*std_return, color='g', linestyle='--', label=f'Mean + 2σ')
plt.axvline(x=mean_return - 2*std_return, color='g', linestyle='--', label=f'Mean - 2σ')
plt.title('Density Plot of Daily Returns')
plt.xlabel('Return')
plt.legend()

plt.tight_layout()
plt.show()

# Print statistical findings
print(f"Average daily return: {mean_return:.4f} ({mean_return*100:.2f}%)")
print(f"Standard deviation of returns: {std_return:.4f} ({std_return*100:.2f}%)")
print(f"Probability of positive return: {prob_positive_return:.4f} ({prob_positive_return*100:.2f}%)")
print(f"Probability of negative return: {prob_negative_return:.4f} ({prob_negative_return*100:.2f}%)")

import xgboost as xgb
import time
from sklearn.model_selection import train_test_split


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE value (percentage)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


seed = 0

xgb_params = {
    # 'objective': 'reg:squarederror',
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'random_state': seed,
    # 'early_stopping_rounds':50
}

def run_model(xgb_params, data):
    print("\nTraining XGBoost model...")
    xgb_model = xgb.XGBRegressor(**xgb_params)

    # Start timing the training
    training_start_time = time.time()

    results_log = {}
    feature_importances = {}
    X, y = data.drop(columns=['target_ticker']), data['target_ticker']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    # Train the model
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )

    predictions = xgb_model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    # mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)

    results_log = {'mes': mse, 'rmse': rmse, 'mape': mape}
    feature_importances = xgb_model.feature_importances_


    training_time = time.time() - training_start_time
    print(f"\nCompleted in {training_time:.2f} seconds")
    
    return results_log, feature_importances

def split_by_date(df, date):
    """
    Split the DataFrame into two parts based on a given date.
    
    Args:
        df: DataFrame to split
        date: Date to split on (datetime object)
    
    Returns:
        Two DataFrames: one before the date and one after
    """
    date = pd.to_datetime(date)
    before_date = df.loc[df.index < date]
    after_date = df.loc[df.index >= date]
    return before_date, after_date

def run_model(xgb_params, data):
    print("\nTraining XGBoost model...")
    xgb_model = xgb.XGBRegressor(**xgb_params)
    
    # Start timing the training
    training_start_time = time.time()
    
    results_log = {}
    feature_importances = {}
    
    # Add date-based features
    data = add_date_features(data)
    
    # Split data by date
    train, test = split_by_date(data, date='2025-01-28')
    X_train, y_train = train.drop(columns=['target_ticker']), train['target_ticker']
    X_test, y_test = test.drop(columns=['target_ticker']), test['target_ticker']
    
    # Train the model with early stopping
    print(f"Training with {X_train.shape[1]} features, including date-based features")
    
    xgb_model.fit(
        X_train,
        y_train,
        verbose=100
    )
    
    # Make predictions
    predictions = xgb_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, predictions)
    
    results_log = {'mse': mse, 'rmse': rmse, 'mape': mape}
    feature_importances = xgb_model.feature_importances_
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, predictions, label='Predicted')
    plt.title('XGBoost: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    training_time = time.time() - training_start_time
    print(f"\nCompleted in {training_time:.2f} seconds")
    
    return results_log, feature_importances

direct_results, direct_feature_importances = run_model(xgb_params, final_merged_df)

# Calculate average results across all windows
avg_mse = np.mean(direct_results['mes'])
avg_rmse = np.mean(direct_results['rmse'])
# avg_mae = np.mean(direct_results)
avg_mape = np.mean(direct_results['mape'])
print("\nAverage Results Across All Windows:")
print(f"Average MSE: {avg_mse:.2f}")
print(f"Average RMSE: {avg_rmse:.2f}")
# print(f"Average MAE: {avg_mae:.2f}")
print(f"Average MAPE: {avg_mape:.2f}%")

def create_features(df, target_col, lag_periods=21, date='2025-02-28'):
    """
    Create lag features and date-based features for time series forecasting.

    Args:
        df (DataFrame): Dataframe containing time series data
        target_col (str): Target column to create lags for
        lag_periods (int): Number of lag periods to create
        date (str): Date to split the data on

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Create a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Add date-based features
    data = add_date_features(data)
    
    # Split the data by date
    train_df, test_df = split_by_date(data, date)
    
    # Make explicit copies to avoid the SettingWithCopyWarning
    train = train_df.copy()
    test = test_df.copy()
    
    # Create lag features using .loc to avoid the warning
    for lag in range(1, lag_periods + 1):
        train.loc[:, f'lag_{lag}'] = train[target_col].shift(lag)

    # Add rolling statistics
    for window in [7, 14, 30]:
        train.loc[:, f'rolling_mean_{window}'] = train[target_col].rolling(window=window).mean()
        train.loc[:, f'rolling_std_{window}'] = train[target_col].rolling(window=window).std()
        
        # Add price momentum features
        train.loc[:, f'price_momentum_{window}'] = train[target_col].pct_change(periods=window)
        
    # Drop NaN values that result from shifting and rolling operations
    train = train.dropna()

    # Create lag columns in test set filled with NaN
    for lag in range(1, lag_periods + 1):
        test.loc[:, f'lag_{lag}'] = np.nan
        
    # Create rolling columns in test set filled with NaN
    for window in [7, 14, 30]:
        test.loc[:, f'rolling_mean_{window}'] = np.nan
        test.loc[:, f'rolling_std_{window}'] = np.nan
        test.loc[:, f'price_momentum_{window}'] = np.nan

    # Separate features and target
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]
    
    print(f"Training with {X_train.shape[1]} features, including date-based and time series features")

    return X_train, X_test, y_train, y_test


def recursive_forecast(model, data):
    """
    Make recursive forecasts using an XGBoost model.

    This function makes multi-step forecasts by using each prediction as
    an input feature for the next prediction (recursive approach).

    Args:
        model (XGBRegressor): Trained XGBoost model
        X_test (DataFrame): Test data with features
        last_train_record (DataFrame): Last record from training data
        last_train_record_tsd (float): Last observed target value

    Returns:
        list: Forecasted values for the entire forecast horizon
    """
    X_train, X_test, y_train, y_test = data
    # Array to store forecasts
    forecasts = []
    
    # Get the lag feature names and sort them numerically
    lag_cols = [col for col in X_test.columns if 'lag_' in col]
    lag_cols.sort(key=lambda x: int(x.split('_')[1]))
    
    # Create a dictionary to store lag values (will be updated with each prediction)
    lag_values = {}
    
    # Initialize lag values from the last training record
    for i, col in enumerate(lag_cols):
        if i == 0:  # The most recent lag needs the last known target value
            lag_values[col] = y_train.values[-1]
        else:
            # Get values from the last training record for all other lags
            lag_values[col] = X_train.iloc[-1][lag_cols[i-1]]
    
    # Make predictions for each time step in the test set
    for i in range(len(X_test)):
        # Create a copy of the current test row
        curr_row = X_test.iloc[i:i+1].copy()
        
        # Update the lag features with our values
        for col in lag_cols:
            curr_row[col] = lag_values[col]
        
        # Make the prediction
        pred = model.predict(curr_row)[0]
        forecasts.append(pred)
        
        # Update lag values for the next step
        # Shift all lags one position (lag_1 becomes lag_2, etc.)
        for j in range(len(lag_cols)-1, 0, -1):
            lag_values[lag_cols[j]] = lag_values[lag_cols[j-1]]
        
        # The newest prediction becomes lag_1
        lag_values[lag_cols[0]] = pred
    
    return forecasts
def run_recursive(xgb_params, data):
    print("\nPerforming recursive forecasting...")

    # Start timing recursive forecasting
    recursive_start_time = time.time()
    results_log = {}
    
    xgb_model = xgb.XGBRegressor(**xgb_params)
    X_train, X_test, y_train, y_test = create_features(data, 'target_ticker', lag_periods=21, date='2025-01-28')
    
    # Train the model
    xgb_model.fit(
        X_train,
        y_train,
        verbose=False
    )
    try:
        model_path = 'models/APPL_xgb_recursive_model.json'
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        xgb_model.save_model(model_path)
        print(f"Model saved to: {model_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
    # Get feature names from X_train
    feature_names = X_train.columns.tolist()
    
    # Create a dictionary mapping feature names to their importance scores
    feature_importances = {name: importance for name, importance in 
                          zip(feature_names, xgb_model.feature_importances_)}
    
    # Make recursive forecasts
    recursive_preds = recursive_forecast(xgb_model, (X_train, X_test, y_train, y_test))

    # Calculate metrics
    mse = mean_squared_error(y_test, recursive_preds)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, recursive_preds)

    results_log = {'mes': mse, 'rmse': rmse, 'mape': mape}
    
    recursive_time = time.time() - recursive_start_time
    print(f"Recursive forecasting completed in {recursive_time:.2f} seconds")
    
    return results_log, feature_importances


results_log, feature_importances = run_recursive(xgb_params, final_merged_df)

# Calculate average results across all windows
rec_mse = np.mean(results_log['mes'])
rec_rmse = np.mean(results_log['rmse'])
# rec_avg_mae = np.mean([result['mae'] for result in recursive_results.values()])
rec_mape = np.mean(results_log['mape'])
print("\nAverage Results Across All Windows:")
print(f"Average MSE: {rec_mse:.2f}")
print(f"Average RMSE: {rec_rmse:.2f}")
# print(f"Average MAE: {rec_avg_mae:.2f}")
print(f"Average MAPE: {rec_mape:.2f}%")

# Extract feature names and their importance values from the dictionary
features = list(feature_importances.keys())
importances = list(feature_importances.values())

# Create a DataFrame to store feature importances
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)


# Get only the top 5 features
top5_importance_df = importance_df.head(5)

# Plot the feature importances (only top 5)
plt.figure(figsize=(14, 6))
sns.barplot(x='Importance', y='Feature', data=top5_importance_df, palette='viridis')
plt.title('Top 5 Features for Stock Price Prediction', fontsize=16)
plt.xlabel('Importance Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()

# Add values to the bars
for i, v in enumerate(top5_importance_df['Importance']):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center')

plt.show()

# Print the top 5 most important features
print("Top 5 Most Important Features:")
print(top5_importance_df)

def add_date_features(df):
    """
    Add calendar-based features to the DataFrame based on the index date.
    
    Args:
        df (DataFrame): DataFrame with datetime index
        
    Returns:
        DataFrame: Original DataFrame with additional date features
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Extract basic date components
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    # Boolean flags for specific date characteristics
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # 1 for weekend, 0 for weekday
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    df['is_year_start'] = df.index.is_year_start.astype(int)
    df['is_year_end'] = df.index.is_year_end.astype(int)
    
    # First/last days of the week (Monday=0, Sunday=6)
    df['is_week_start'] = (df['day_of_week'] == 0).astype(int)  # Monday
    df['is_week_end'] = (df['day_of_week'] == 6).astype(int)    # Sunday
    
    return df

final_merged_df = add_date_features(final_merged_df)

# Display the first few rows of the DataFrame with the new date features
final_merged_df.head()