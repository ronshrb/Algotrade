import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import time
import datetime as dt
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Import our MPT tool
from mpt_interactive import display_mpt_page

# Set page configuration
st.set_page_config(
    page_title="Stock Analysis & Forecasting Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0277BD;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        color: #424242;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">Stock Analysis & Forecasting Tool</p>', unsafe_allow_html=True)

# Define leading tickers
leading_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Define functions
def download_data(tickers, start='2020-01-01', end=None):
    """
    Download stock data from Yahoo Finance for multiple tickers.
    """
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
        
    stock_data = pd.DataFrame()
    for name, ticker in tickers.items():
        data = yf.download(ticker, start=start, end=end)
        if len(data) > 0:
            stock_data[name] = data['Close']
    
    return stock_data

def get_and_prepare_unemployment_data():
    """
    Download U.S. unemployment rate data (UNRATE) from FRED,
    resample to daily frequency, and interpolate linearly.
    """
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
        response = pd.read_csv(url, parse_dates=['DATE'], index_col='DATE')
        response.columns = ['UNRATE']
        response['Unemployment Rate'] = response['UNRATE']
        return response
    except Exception as e:
        st.error(f"Error fetching unemployment data: {e}")
        return pd.DataFrame(columns=['Unemployment Rate'])

def add_technical_indicators(df, target_ticker_col):
    """
    Add RSI and MACD indicators to the DataFrame.
    """
    # Initialize the RSI and MACD indicators
    rsi_indicator = RSIIndicator(df[target_ticker_col])
    macd_indicator = MACD(df[target_ticker_col])

    # Calculate RSI and MACD
    df['RSI'] = rsi_indicator.rsi()
    df['MACD'] = macd_indicator.macd()  # Moving Average Convergence Divergence
    df['MACD Signal'] = macd_indicator.macd_signal()
    df['MACD Histogram'] = macd_indicator.macd_diff()

    return df

def prepare_data(ticker_selection):
    """
    Prepare and clean data for the selected ticker.
    """
    # Create tickers dictionary
    tickers = {
        'target_ticker': ticker_selection,
        'Nasdaq_100': 'QQQ',
        'Finance_Sector': 'XLF',
        'Europe_Stocks': 'FEZ',
        'Gold_ETF': 'GLD',
    }
    
    # Download data
    with st.spinner('Downloading market data...'):
        start_date = '2020-01-01'
        tickers_data = download_data(tickers, start=start_date)
    
    if tickers_data.empty:
        st.error("Failed to download market data. Please try again.")
        return None
        
    # Add technical indicators
    with st.spinner('Calculating technical indicators...'):
        tickers_data = add_technical_indicators(tickers_data, 'target_ticker')
    
    # Get unemployment data
    with st.spinner('Fetching economic data...'):
        employment_data = get_and_prepare_unemployment_data()
    
    # Merge datasets if unemployment data is available
    if not employment_data.empty:
        # First, add year and month columns to both dataframes
        tickers_data['year'] = tickers_data.index.year
        tickers_data['month'] = tickers_data.index.month
        employment_data['year'] = employment_data.index.year
        employment_data['month'] = employment_data.index.month

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
    else:
        merged_df = tickers_data
    
    # Handle missing values
    merged_df = merged_df.dropna()
    
    return merged_df

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def split_by_date(df, date):
    """
    Split the DataFrame into two parts based on a given date.
    """
    date = pd.to_datetime(date)
    before_date = df.loc[df.index < date]
    after_date = df.loc[df.index >= date]
    return before_date, after_date

def create_features(df, target_col, lag_periods=21, future_periods=0):
    """
    Create lag features for time series forecasting.
    Uses all data for training and optionally creates empty rows for future predictions.
    
    Args:
        df (DataFrame): Input data
        target_col (str): Target column name
        lag_periods (int): Number of lag periods to create
        future_periods (int): Number of future periods to create placeholders for
        
    Returns:
        tuple: X_train, X_future, y_train, train_df
    """
    # Create a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Create lag features in the training data
    for lag in range(1, lag_periods + 1):
        data.loc[:, f'lag_{lag}'] = data[target_col].shift(lag)

    # Drop NaN values that result from shifting
    train_df = data.dropna().copy()
    
    # Separate features and target for training
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    # Create placeholder for future predictions if requested
    X_future = None
    if future_periods > 0:
        # Create a DataFrame for future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                     periods=future_periods)
        X_future = pd.DataFrame(index=future_dates, columns=X_train.columns)
        
        # Initialize with the last known values for non-lag features
        for col in X_train.columns:
            if 'lag_' not in col:
                X_future[col] = X_train.iloc[-1][col]
    
    return X_train, X_future, y_train, train_df

def recursive_forecast(model, data):
    """
    Make recursive forecasts using an XGBoost model.

    This function makes multi-step forecasts by using each prediction as
    an input feature for the next prediction (recursive approach).

    Args:
        model (XGBRegressor): Trained XGBoost model
        data (tuple): Tuple containing (X_train, X_test, y_train, y_test)

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


def recursive_forecast_future(model, X_train, y_train, forecast_horizon=30, last_known_data=None):
    """
    Make recursive forecasts for future dates using an XGBoost model.
    
    Args:
        model (XGBRegressor): Trained XGBoost model
        X_train (DataFrame): Training features
        y_train (Series): Training target values
        forecast_horizon (int): Number of days to forecast
        last_known_data (list): Last known prices for initialization
        
    Returns:
        list: Forecasted values for the future horizon
    """
    # Array to store forecasts
    forecasts = []
    
    # Get the lag feature names and sort them numerically
    lag_cols = [col for col in X_train.columns if 'lag_' in col]
    lag_cols.sort(key=lambda x: int(x.split('_')[1]))
    
    # Create a dictionary to store lag values (will be updated with each prediction)
    lag_values = {}
    
    # Initialize lag values from the last training record or provided data
    if last_known_data is not None:
        # Use the provided data for initialization
        for i, col in enumerate(lag_cols):
            lag_num = int(col.split('_')[1])
            if lag_num <= len(last_known_data):
                lag_values[col] = last_known_data[-lag_num]
    else:
        # Initialize from training data
        for i, col in enumerate(lag_cols):
            if i == 0:  # The most recent lag needs the last known target value
                lag_values[col] = y_train.values[-1]
            else:
                # Get values from the last training record for all other lags
                lag_values[col] = X_train.iloc[-1][lag_cols[i-1]]
    
    # Make predictions for the forecast horizon
    for i in range(forecast_horizon):
        # Create a new row with all features
        curr_row = pd.DataFrame(index=[i], columns=X_train.columns)
        
        # Fill the row with the features from the last training example for non-lag columns
        for col in X_train.columns:
            if 'lag_' not in col:
                curr_row[col] = X_train.iloc[-1][col]
        
        # Update lag features with our values
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


def run_recursive(xgb_params, data, ticker_symbol='AAPL', save_model=True, forecast_horizon=30):
    """
    Train a recursive XGBoost model and make predictions.
    
    Args:
        xgb_params (dict): XGBoost model parameters
        data (DataFrame): Input data
        ticker_symbol (str): Stock ticker symbol
        save_model (bool): Whether to save the trained model
        forecast_horizon (int): Number of days to forecast
        
    Returns:
        tuple: Results log, feature importances, model, and data
    """    
    # Create features with lag values for all data
    X_train, _, y_train, train_df = create_features(data, 'target_ticker', lag_periods=21)
    
    # Check if model already exists
    model_path = f'models/{ticker_symbol}_xgb_recursive_model.json'
    model_exists = os.path.exists(model_path)
    
    if model_exists:
        # Load existing model
        st.info(f"Loading existing recursive model for {ticker_symbol}...")
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(model_path)
    else:
        # Train new model on all data
        st.info(f"Training new recursive model for {ticker_symbol}...")
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train, verbose=False)
        
        if save_model:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                # Save the model
                xgb_model.save_model(model_path)
                st.success(f"Model saved to: {model_path}")
                
            except Exception as e:
                st.error(f"Error saving model: {e}")
    
    # Get feature names from X_train
    feature_names = X_train.columns.tolist()
    
    # Create a dictionary mapping feature names to their importance scores
    feature_importances = {name: importance for name, importance in 
                          zip(feature_names, xgb_model.feature_importances_)}
    
    # Get the last known prices for recursive forecasting initialization
    last_known_prices = data['target_ticker'][-21:].values.tolist()
    
    # Generate future forecasts
    future_predictions = recursive_forecast_future(
        model=xgb_model,
        X_train=X_train,
        y_train=y_train,
        forecast_horizon=forecast_horizon,
        last_known_data=last_known_prices
    )
    
    # Create future dates for the forecast
    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_horizon)
    
    # Create a results log with the forecasts
    results_log = {
        'forecast_dates': future_dates,
        'forecast_values': future_predictions,
        'last_date': data.index[-1],
        'last_price': data['target_ticker'].iloc[-1]
    }
    
    return results_log, feature_importances, xgb_model, X_train, y_train

def standard_forecast(model, X_test):
    """
    Make standard single-step forecasts using the model.
    """
    return model.predict(X_test)


def load_or_train_model(ticker_symbol, model_type, xgb_params, X_train, y_train):
    """
    Load an existing model or train a new one if it doesn't exist.
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        model_type (str): Type of model ('standard' or 'recursive')
        xgb_params (dict): XGBoost model parameters
        X_train (DataFrame): Training features
        y_train (Series): Training target values
        
    Returns:
        XGBRegressor: Loaded or trained model
    """
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Construct the model filename based on ticker and model type
    model_suffix = 'standard' if model_type == 'standard' else 'recursive'
    model_path = os.path.join(model_dir, f'{ticker_symbol}_{model_suffix}_model.json')
    
    # Check if model exists
    if os.path.exists(model_path):
        st.info(f"Loading existing {model_type} model for {ticker_symbol}...")
        # Load the model
        model = xgb.XGBRegressor()
        model.load_model(model_path)
    else:
        st.info(f"Training new {model_type} model for {ticker_symbol}...")
        # Train a new model
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train, verbose=False)
        
        # Save the model
        try:
            model.save_model(model_path)
            st.success(f"Model saved to: {model_path}")
        except Exception as e:
            st.error(f"Error saving model: {e}")
    
    return model

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Visualization", "🔮 Forecasting", "📈 Portfolio Optimization"])

with tab1:
    st.markdown('<p class="sub-header">Stock Data Visualization</p>', unsafe_allow_html=True)
    
    # Sidebar for Data Visualization tab
    with st.sidebar:
        st.header("Data Settings")
        ticker_selection = st.selectbox('Select Stock Ticker:', leading_tickers)
        start_date = st.date_input('Start Date:', dt.date(2020, 1, 1))
        end_date = st.date_input('End Date:', dt.datetime.now().date())
        
    # Prepare data
    tickers_dict = {
        'target_ticker': ticker_selection,
        'Nasdaq_100': 'QQQ',
        'Finance_Sector': 'XLF',
        'Europe_Stocks': 'FEZ',
        'Gold_ETF': 'GLD',
    }
    
    df = download_data(tickers_dict, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    if not df.empty:
        # Add technical indicators
        df = add_technical_indicators(df, 'target_ticker')
        
        # Price Charts
        st.markdown('<p class="sub-header">Price Chart</p>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['target_ticker'], label=f'{ticker_selection} Close Price')
        ax.set_title(f'{ticker_selection} Stock Price', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        
        # Technical Indicators
        st.markdown('<p class="sub-header">Technical Indicators</p>', unsafe_allow_html=True)
        tech_indicator = st.selectbox('Select Technical Indicator:', 
                                    ['RSI', 'MACD'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if tech_indicator == 'RSI':
            ax.plot(df.index, df['RSI'], color='purple')
            ax.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax.set_title('Relative Strength Index (RSI)', fontsize=16)
            ax.set_ylabel('RSI Value', fontsize=12)
            ax.set_ylim(0, 100)
            ax.fill_between(df.index, y1=70, y2=100, color='red', alpha=0.1)
            ax.fill_between(df.index, y1=0, y2=30, color='green', alpha=0.1)
        
        else:  # MACD
            ax.plot(df.index, df['MACD'], label='MACD Line', color='blue')
            ax.plot(df.index, df['MACD Signal'], label='Signal Line', color='red')
            ax.bar(df.index, df['MACD Histogram'], label='Histogram', color=np.where(df['MACD Histogram'] >= 0, 'green', 'red'), width=1)
            ax.set_title('Moving Average Convergence Divergence (MACD)', fontsize=16)
            ax.legend()
        
        ax.set_xlabel('Date', fontsize=12)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Correlation Matrix
        st.markdown('<p class="sub-header">Correlation Matrix</p>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df.corr()
        mask = np.triu(corr_matrix)
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5,
            mask=mask,
            vmin=-1, vmax=1,
            ax=ax
        )
        ax.set_title('Correlation Matrix', fontsize=16)
        st.pyplot(fig)
        
        # Daily Returns Analysis
        st.markdown('<p class="sub-header">Daily Returns Analysis</p>', unsafe_allow_html=True)
        
        returns = df['target_ticker'].pct_change().dropna()
        
        # Statistics
        mean_return = returns.mean()
        std_return = returns.std()
        prob_positive = (returns > 0).mean()
        prob_negative = (returns < 0).mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Daily Return", f"{mean_return*100:.2f}%")
        with col2:
            st.metric("Standard Deviation", f"{std_return*100:.2f}%")
        with col3:
            st.metric("Positive Days", f"{prob_positive*100:.1f}%")
        with col4:
            st.metric("Negative Days", f"{prob_negative*100:.1f}%")
        
        # Returns Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        returns.hist(bins=50, alpha=0.7, ax=ax1)
        ax1.axvline(x=mean_return, color='r', linestyle='--', label=f'Mean: {mean_return:.4f}')
        ax1.set_title('Distribution of Daily Returns', fontsize=14)
        ax1.set_xlabel('Return', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        
        # Density plot
        sns.kdeplot(returns, fill=True, ax=ax2)
        ax2.axvline(x=mean_return, color='r', linestyle='--', label=f'Mean: {mean_return:.4f}')
        ax2.axvline(x=mean_return + 2*std_return, color='g', linestyle='--', label=f'Mean + 2σ')
        ax2.axvline(x=mean_return - 2*std_return, color='g', linestyle='--', label=f'Mean - 2σ')
        ax2.set_title('Density Plot of Daily Returns', fontsize=14)
        ax2.set_xlabel('Return', fontsize=12)
        ax2.legend()
        
        st.pyplot(fig)
        
        # Data Table
        st.markdown('<p class="sub-header">Price Data</p>', unsafe_allow_html=True)
        st.dataframe(df)
    else:
        st.error("Failed to download data. Please check the ticker symbol and try again.")

with tab2:
    st.markdown('<p class="sub-header">Stock Price Forecasting</p>', unsafe_allow_html=True)
    
    # Sidebar for Forecasting tab
    with st.sidebar:
        st.header("Forecasting Settings")
        ticker_forecast = st.selectbox('Select Stock for Forecasting:', leading_tickers, key='forecast_ticker')
        model_type = st.selectbox('Select Forecasting Model:', ['XGBoost Standard', 'XGBoost Recursive'])
        forecast_horizon = st.slider('Forecast Horizon (Days):', 5, 30, 14)
    
    # Button to trigger forecasting
    if st.button('Generate Forecast'):
        # Prepare data
        with st.spinner('Preparing data for forecasting...'):
            data = prepare_data(ticker_forecast)
            
            if data is not None:
                # Define XGBoost parameters
                xgb_params = {
                    'n_estimators': 1000,
                    'learning_rate': 0.01,
                    'max_depth': 6,
                    'random_state': 42,
                }
                
                # Get the last known values for display
                last_known_date = data.index[-1]
                last_known_price = data['target_ticker'].iloc[-1]
                
                if model_type == 'XGBoost Standard':
                    with st.spinner('Training standard XGBoost model...'):
                        # Standard XGBoost approach
                        # Prepare features (all data)
                        X_train = data.drop(columns=['target_ticker'])
                        y_train = data['target_ticker']
    
                        # Load or train the model
                        model = load_or_train_model(
                            ticker_symbol=ticker_forecast,
                            model_type='standard',
                            xgb_params=xgb_params,
                            X_train=X_train,
                            y_train=y_train
                        )
                        
                        # Create future dates for predictions
                        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_horizon)
                        
                        # Create a DataFrame with the same features as X_train for future predictions
                        future_features = pd.DataFrame(index=future_dates, columns=X_train.columns)
                        
                        # Fill with the last available values (we assume these features stay constant)
                        for col in future_features.columns:
                            future_features[col] = X_train.iloc[-1][col]
                        
                        # Make future predictions
                        future_predictions = model.predict(future_features)
                    
                    # Display results
                    st.success('Forecast generated successfully!')
                    
                    # Create a plot with historical and predicted values
                    fig, ax = plt.subplots(figsize=(16, 8))
                    
                    # Plot historical data (show last 100 days for better visibility)
                    ax.plot(data.index[-100:], data['target_ticker'][-100:], label='Historical Data', color='blue')
                    
                    # Plot future predictions
                    ax.plot(future_dates, future_predictions, label='Future Forecast', color='red', marker='o')
                    
                    # Add a vertical line to mark the present
                    ax.axvline(x=data.index[-1], color='black', linestyle='--', alpha=0.7)
                    ax.text(data.index[-1], data['target_ticker'].min(), 'Present', ha='right', va='bottom', rotation=90)
                    
                    # Set plot properties
                    ax.set_title(f'{ticker_forecast} Stock Price Forecast', fontsize=16)
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Price ($)', fontsize=12)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # Display last known price
                    st.metric("Last Known Price", f"${last_known_price:.2f} ({last_known_date.strftime('%Y-%m-%d')})")
                    
                    # Display forecast table
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_predictions
                    }).set_index('Date')
                    
                    st.markdown('<p class="sub-header">Forecast Values</p>', unsafe_allow_html=True)
                    st.dataframe(forecast_df.style.format({'Predicted Price': '${:.2f}'}))
                    
                else:  # XGBoost Recursive
                    # Recursive XGBoost approach using the run_recursive function
                    with st.spinner("Running recursive forecasting..."):
                        # Run the recursive model
                        results_log, feature_importances, model, X_train, y_train = run_recursive(
                            xgb_params=xgb_params,
                            data=data,
                            ticker_symbol=ticker_forecast,
                            save_model=True,
                            forecast_horizon=forecast_horizon
                        )
                        
                        # Get future dates and predictions from results
                        future_dates = results_log['forecast_dates']
                        future_predictions = results_log['forecast_values']
                    
                    # Display results
                    st.success('Recursive forecast generated successfully!')
                    
                    # Create a plot with historical and predicted values
                    fig, ax = plt.subplots(figsize=(16, 8))
                    
                    # Plot historical data (show last 100 days for better visibility)
                    ax.plot(data.index[-100:], data['target_ticker'][-100:], label='Historical Data', color='blue')
                    
                    # Plot future predictions
                    ax.plot(future_dates, future_predictions, label='Future Forecast (Recursive)', color='red', marker='o')
                    
                    # Add a vertical line to mark the present
                    ax.axvline(x=data.index[-1], color='black', linestyle='--', alpha=0.7)
                    ax.text(data.index[-1], data['target_ticker'].min(), 'Present', ha='right', va='bottom', rotation=90)
                    
                    # Set plot properties
                    ax.set_title(f'{ticker_forecast} Stock Price Forecast (Recursive)', fontsize=16)
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Price ($)', fontsize=12)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # Display last known price
                    st.metric("Last Known Price", f"${last_known_price:.2f} ({last_known_date.strftime('%Y-%m-%d')})")
                    
                    # Display forecast table
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_predictions
                    }).set_index('Date')
                    
                    st.markdown('<p class="sub-header">Forecast Values</p>', unsafe_allow_html=True)
                    st.dataframe(forecast_df.style.format({'Predicted Price': '${:.2f}'}))
                    
                    # Display feature importance
                    st.markdown('<p class="sub-header">Feature Importance</p>', unsafe_allow_html=True)
                    
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
                    
                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=top5_importance_df, palette='viridis', ax=ax)
                    ax.set_title('Top 5 Features for Stock Price Prediction', fontsize=14)
                    ax.set_xlabel('Importance Score', fontsize=12)
                    ax.set_ylabel('Feature', fontsize=12)
                    ax.grid(axis='x', linestyle='--', alpha=0.6)
                    
                    # Add values to the bars
                    for i, v in enumerate(top5_importance_df['Importance']):
                        ax.text(v + 0.01, i, f'{v:.4f}', va='center')
                    
                    st.pyplot(fig)

# Tab 3 - Portfolio Optimization
with tab3:
    display_mpt_page()

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; color: #777;">
    <p>Algorithmic Trading Analysis & Forecasting Tool</p>
</div>
""", unsafe_allow_html=True)
