# Algotrade - Stock Analysis, Forecasting & Portfolio Optimization Tool

This is a web application built using Streamlit that allows you to analyze stocks, forecast prices, and optimize investment portfolios using algorithmic trading techniques.

## Try the Web App Online

You can use the Modern Portfolio Theory (MPT) Optimizer directly in your browser:

[Launch the Algotrade MPT Web App](https://algotrade-mpt-tool.streamlit.app/)

## Features

### 1. Data Visualization Tab
- Select any ticker from leading tech stocks
- View price charts and technical indicators
- Analyze correlation between different market factors
- View daily returns distribution and statistics

### 2. Forecasting Tab
- Choose between standard XGBoost or recursive XGBoost models
- Select a ticker to forecast
- Specify the forecast horizon (number of days)
- View both historical and forecasted prices
- See forecast values in a table format

### 3. Portfolio Optimization Tab (Modern Portfolio Theory)
- Create diversified portfolios using MPT principles
- Visualize the efficient frontier
- Find optimal portfolio allocations for different risk-return profiles
- Analyze asset correlation for effective diversification

### 4. Investment Simulator Tab
- Simulate how an initial investment would have grown over time in each optimal portfolio
- No need to select a forecast start or end date—simulation uses the full available period
- Visualize portfolio value evolution with interactive charts
- Compare different optimal strategies (Max Returns, Min Volatility, Max Sharpe Ratio)

## Standalone Modern Portfolio Theory (MPT) Tool

For users specifically interested in portfolio optimization, a standalone version of the MPT tool is available:

### Features of the Standalone MPT Tool:
- Input your own selection of stocks/ETFs
- Set custom date ranges for historical data analysis
- Generate the efficient frontier with thousands of simulated portfolios
- Identify minimum volatility and maximum Sharpe ratio portfolios
- View optimal asset allocations visually through interactive charts
- Download portfolio simulation results for further analysis
- Analyze asset correlation matrix


For detailed information about Modern Portfolio Theory and how to use the tool effectively, see the [MPT_GUIDE.md](MPT_GUIDE.md) file.
- Visualize the Efficient Frontier with interactive charts
- Find optimal portfolios (minimum volatility, maximum Sharpe ratio)
- See asset allocations for optimal portfolios
- Analyze correlation between assets in your portfolio
- Download simulation results for further analysis

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run MPT_tool.py
```

## Usage

1. Select the desired tab (Data Visualization, Forecasting, or Portfolio Optimization)
2. Choose a ticker symbol and adjust other parameters in the sidebar
3. For forecasting, click the "Generate Forecast" button after selecting your preferred model and settings
4. For portfolio optimization, enter stock symbols, select date range, and click "Generate Portfolio Analysis"
5. Explore the visualizations and analysis

## Models

1. **Linear Regression** - Uses a standard linear regression model for price forecasting
2. **Random Forest** - Uses a random forest regressor for price forecasting

## Modern Portfolio Theory (MPT)

The Portfolio Optimization tab implements Modern Portfolio Theory, which helps investors:

- Maximize returns for a given level of risk
- Minimize risk for a given level of return
- Find the optimal balance between risk and return (Sharpe ratio)
- Create diversified portfolios that reduce overall risk



## Requirements

- Python 3.8+
- See requirements.txt for all dependencies

