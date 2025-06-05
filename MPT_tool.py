import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet



def fetch_stock_tickers():
    """
    Fetches stock tickers from major indices and returns them as a DataFrame
    """

    
    all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'TSLA', 'NVDA', 'AMD', 'INTC']
    

    return all_tickers


def price_forecast(stock, start_date, end_date, record_percentage_to_predict):
    """

    """
    df = yf.download(stock, start=start_date, end=end_date)

    df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100.0
    df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

    forecast_col = 'Close'
    # forecast_col = 'Adj. Close'
    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(record_percentage_to_predict * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)
    # print(df.head())


    X = np.array(df.drop(['label'], axis=1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    df.dropna(inplace=True)
    df['Close'].plot()
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=record_percentage_to_predict)
    # clf =  svm.SVR() # #LinearRegression()
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    # print(confidence)
    forecast_set = clf.predict(X_lately)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    # df['Close'].plot()
    # df['Forecast'].plot()
    # plt.legend(loc=4)
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.show()

    col = df['Forecast']
    col = col.dropna()
    return col

def price_forecast_rf(stock, start_date, end_date, record_percentage_to_predict):
    """

    """
    df = yf.download(stock, start=start_date, end=end_date)

    df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100.0
    df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

    forecast_col = 'Close'
    # forecast_col = 'Adj. Close'
    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(record_percentage_to_predict * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)
    print(df.head())


    X = np.array(df.drop(['label'], axis=1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    df.dropna(inplace=True)
    df['Close'].plot()
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=record_percentage_to_predict)
    # clf =  svm.SVR() # #LinearRegression()
    clf = RandomForestRegressor(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    # print(confidence)
    forecast_set = clf.predict(X_lately)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    col = df['Forecast']
    col = col.dropna()
    return col

def random_forest(start_date, end_date, Num_porSimulation, selected, record_percentage_to_predict):

    #yf.pdr_override()
    frame = {}
    for stock in selected:
        # call price_forecast on each stock and get prediction for it's prices
        price = price_forecast_rf(stock, start_date, end_date, record_percentage_to_predict)
        frame[stock] = price
    #frame.to_csv('1.csv')
    table = pd.DataFrame(frame)
    plt.plot(pd.DataFrame(frame))
    pd.DataFrame(frame).to_csv('Out.csv')

    returns_daily = table.pct_change()
    returns_daily.to_csv('Out1.csv')
    returns_annual = ((1 + returns_daily.mean()) ** 254) - 1
    returns_annual.to_csv('Out2.csv')

    # get daily and covariance of returns of the stock
    cov_daily = returns_daily.cov()
    cov_annual = cov_daily * 250

    # empty lists to store returns, volatility and weights of imiginary portfolios
    port_returns = []
    port_volatility = []
    sharpe_ratio = []
    stock_weights = []

    # set the number of combinations for imaginary portfolios
    num_assets = len(selected)
    num_portfolios = Num_porSimulation  # Change porfolio numbers here

    # set random seed for reproduction's sake
    np.random.seed(101)

    # populate the empty lists with each portfolios returns,risk and weights
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        sharpe = returns / volatility
        sharpe_ratio.append(sharpe)
        port_returns.append(returns * 100)
        port_volatility.append(volatility * 100)
        stock_weights.append(weights)

    # a dictionary for Returns and Risk values of each portfolio
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility,
                 'Sharpe Ratio': sharpe_ratio}

    # extend original dictionary to accomodate each ticker and weight in the portfolio
    for counter, symbol in enumerate(selected):
        portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]

    # make a nice dataframe of the extended dictionary
    df = pd.DataFrame(portfolio)

    # get better labels for desired arrangement of columns
    column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock + ' Weight' for stock in selected]

    # reorder dataframe columns
    df = df[column_order]

    # plot frontier, max sharpe & min Volatility values with a scatterplot
    # find min Volatility & max sharpe values in the dataframe (df)
    min_volatility = df['Volatility'].min()
    # min_volatility1 = df['Volatility'].min()+1
    max_sharpe = df['Sharpe Ratio'].max()
    max_return = df['Returns'].max()
    max_vol = df['Volatility'].max()
    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    min_variance_port = df.loc[df['Volatility'] == min_volatility]
    max_returns = df.loc[df['Returns'] == max_return]
    max_vols = df.loc[df['Volatility'] == max_vol]

    # plot frontier, max sharpe & min Volatility values with a scatterplot
    plt.clf()
    plt.style.use('seaborn-v0_8-dark')
    ax = df.plot.scatter(
        x='Volatility', y='Returns', c='Sharpe Ratio',
        cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True, alpha=0.8
    )
    plt.scatter(
        x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'],
        c='green', marker='D', s=200, label='Max Sharpe'
    )
    plt.scatter(
        x=min_variance_port['Volatility'], y=min_variance_port['Returns'],
        c='orange', marker='D', s=200, label='Min Volatility'
    )
    plt.scatter(
        x=max_vols['Volatility'], y=max_returns['Returns'],
        c='red', marker='D', s=200, label='Max Return'
    )
    plt.xlabel('Volatility (Std. Deviation) Percentage %')
    plt.ylabel('Expected Returns Percentage %')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.tight_layout()

    # Show plot in Streamlit
    st.subheader("Efficient Frontier")
    with st.expander("What is the Efficient Frontier?"):
        st.markdown("""
    The efficient frontier is a concept from Modern Portfolio Theory. It represents the set of optimal portfolios that offer the highest expected return for a given level of risk (volatility), or the lowest risk for a given level of expected return. 

    - Each point on the frontier is a portfolio that is "efficient"—meaning you can't get a higher return without taking on more risk, or reduce risk without lowering your expected return.
    - Portfolios below the frontier are sub-optimal, because they do not provide enough return for the level of risk.
    - The shape of the frontier helps investors visualize the trade-off between risk and return, and select a portfolio that matches their risk tolerance.

    The chart below shows the simulated portfolios, with the efficient frontier highlighted by the best combinations of risk and return.
    """)
    st.pyplot(plt.gcf())

    # Find indices for optimal portfolios
    red_num = df[df["Returns"] == max_return].index
    yellow_num = df[df['Volatility'] == min_volatility].index
    green_num = df[df['Sharpe Ratio'] == max_sharpe].index

    # Show optimal portfolios in Streamlit
    multseries = pd.Series([1, 1, 1] + [100 for stock in selected],
                        index=['Returns', 'Volatility', 'Sharpe Ratio'] + [stock + ' Weight' for stock in selected])

    # Prepare DataFrame for optimal portfolios
    optimal_rows = []
    optimal_names = []
    if len(red_num) > 0:
        optimal_rows.append(df.loc[red_num[0]].multiply(multseries))
        optimal_names.append("🚩 Max Returns")
    if len(yellow_num) > 0:
        optimal_rows.append(df.loc[yellow_num[0]].multiply(multseries))
        optimal_names.append("🟨 Min Volatility")
    if len(green_num) > 0:
        optimal_rows.append(df.loc[green_num[0]].multiply(multseries))
        optimal_names.append("🟩 Max Sharpe Ratio")

    if optimal_rows:
        optimal_df = pd.DataFrame(optimal_rows, index=optimal_names)
        st.subheader("Optimal Portfolios Table")
        st.dataframe(optimal_df.style.format("{:.2f}"), hide_index=False)
    else:
        st.info("No optimal portfolios found.")

    # Optionally, show the dataframe
    with st.expander("Show All Simulated Portfolios DataFrame"):
        st.dataframe(df.style.format("{:.2f}"), hide_index=True)

#def markovich(start_date, end_date, Num_porSimulation, selected, record_percentage_to_predict):
def markovich(start_date, end_date, Num_porSimulation, selected, record_percentage_to_predict):

    #yf.pdr_override()
    frame = {}
    for stock in selected:
        # call price_forecast on each stock and get prediction for it's prices
        price = price_forecast(stock, start_date, end_date, record_percentage_to_predict)
        frame[stock] = price
    #frame.to_csv('1.csv')
    table = pd.DataFrame(frame)
    plt.plot(pd.DataFrame(frame))
    pd.DataFrame(frame).to_csv('Out.csv')

    returns_daily = table.pct_change()
    returns_daily.to_csv('Out1.csv')
    returns_annual = ((1 + returns_daily.mean()) ** 254) - 1
    returns_annual.to_csv('Out2.csv')

    # get daily and covariance of returns of the stock
    cov_daily = returns_daily.cov()
    cov_annual = cov_daily * 250

    # empty lists to store returns, volatility and weights of imiginary portfolios
    port_returns = []
    port_volatility = []
    sharpe_ratio = []
    stock_weights = []

    # set the number of combinations for imaginary portfolios
    num_assets = len(selected)
    num_portfolios = Num_porSimulation  # Change porfolio numbers here

    # set random seed for reproduction's sake
    np.random.seed(101)

    # populate the empty lists with each portfolios returns,risk and weights
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        sharpe = returns / volatility
        sharpe_ratio.append(sharpe)
        port_returns.append(returns * 100)
        port_volatility.append(volatility * 100)
        stock_weights.append(weights)

    # a dictionary for Returns and Risk values of each portfolio
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility,
                 'Sharpe Ratio': sharpe_ratio}

    # extend original dictionary to accomodate each ticker and weight in the portfolio
    for counter, symbol in enumerate(selected):
        portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]

    # make a nice dataframe of the extended dictionary
    df = pd.DataFrame(portfolio)

    # get better labels for desired arrangement of columns
    column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock + ' Weight' for stock in selected]

    # reorder dataframe columns
    df = df[column_order]

    # plot frontier, max sharpe & min Volatility values with a scatterplot
    # find min Volatility & max sharpe values in the dataframe (df)
    min_volatility = df['Volatility'].min()
    # min_volatility1 = df['Volatility'].min()+1
    max_sharpe = df['Sharpe Ratio'].max()
    max_return = df['Returns'].max()
    max_vol = df['Volatility'].max()
    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    min_variance_port = df.loc[df['Volatility'] == min_volatility]
    max_returns = df.loc[df['Returns'] == max_return]
    max_vols = df.loc[df['Volatility'] == max_vol]

    # Find indices for optimal portfolios
    red_num = df[df["Returns"] == max_return].index
    yellow_num = df[df['Volatility'] == min_volatility].index
    green_num = df[df['Sharpe Ratio'] == max_sharpe].index


    # plot frontier, max sharpe & min Volatility values with a scatterplot
    plt.clf()
    plt.style.use('seaborn-v0_8-dark')
    ax = df.plot.scatter(
        x='Volatility', y='Returns', c='Sharpe Ratio',
        cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True, alpha=0.8
    )
    plt.scatter(
        x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'],
        c='green', marker='D', s=200, label='Max Sharpe'
    )
    plt.scatter(
        x=min_variance_port['Volatility'], y=min_variance_port['Returns'],
        c='orange', marker='D', s=200, label='Min Volatility'
    )
    plt.scatter(
        x=max_vols['Volatility'], y=max_returns['Returns'],
        c='red', marker='D', s=200, label='Max Return'
    )
    plt.xlabel('Volatility (Std. Deviation) Percentage %')
    plt.ylabel('Expected Returns Percentage %')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.tight_layout()

    # Show plot in Streamlit
    st.subheader("Efficient Frontier")
    with st.expander("What is the Efficient Frontier?"):
        st.markdown("""
    The efficient frontier is a concept from Modern Portfolio Theory. It represents the set of optimal portfolios that offer the highest expected return for a given level of risk (volatility), or the lowest risk for a given level of expected return. 

    - Each point on the frontier is a portfolio that is "efficient"—meaning you can't get a higher return without taking on more risk, or reduce risk without lowering your expected return.
    - Portfolios below the frontier are sub-optimal, because they do not provide enough return for the level of risk.
    - The shape of the frontier helps investors visualize the trade-off between risk and return, and select a portfolio that matches their risk tolerance.

    The chart below shows the simulated portfolios, with the efficient frontier highlighted by the best combinations of risk and return.
    """)
    st.pyplot(plt.gcf())



    # Show optimal portfolios in Streamlit
    multseries = pd.Series([1, 1, 1] + [100 for stock in selected],
                        index=['Returns', 'Volatility', 'Sharpe Ratio'] + [stock + ' Weight' for stock in selected])

    # Prepare DataFrame for optimal portfolios
    optimal_rows = []
    optimal_names = []
    if len(red_num) > 0:
        optimal_rows.append(df.loc[red_num[0]].multiply(multseries))
        optimal_names.append("🚩 Max Returns")
    if len(yellow_num) > 0:
        optimal_rows.append(df.loc[yellow_num[0]].multiply(multseries))
        optimal_names.append("🟨 Min Volatility")
    if len(green_num) > 0:
        optimal_rows.append(df.loc[green_num[0]].multiply(multseries))
        optimal_names.append("🟩 Max Sharpe Ratio")

    if optimal_rows:
        optimal_df = pd.DataFrame(optimal_rows, index=optimal_names)
        st.subheader("Optimal Portfolios Table")
        st.dataframe(optimal_df.style.format("{:.2f}"), hide_index=False)
    else:
        st.info("No optimal portfolios found.")

    # Optionally, show the dataframe
    with st.expander("Show All Simulated Portfolios DataFrame"):
        st.dataframe(df.style.format("{:.2f}"), hide_index=True)

def price_forecast_prophet(stock, start_date, end_date, record_percentage_to_predict):
    df = yf.download(stock, start=start_date, end=end_date)

    # Only keep 'Date' and 'Close', ensure it's in the right format
    df = df.reset_index()[['Date', 'Close']].dropna()
    df.columns = ['ds', 'y']  # Prophet requires these exact names
    df['y'] = pd.to_numeric(df['y'], errors='coerce')  # Coerce non-numeric values

    # Drop any rows with NaN after coercion
    df = df.dropna()

    # Forecast horizon
    forecast_out = int(math.ceil(record_percentage_to_predict * len(df)))
    train_df = df[:-forecast_out]

    # Fit the Prophet model
    model = Prophet()
    model.fit(train_df)

    # Create future dataframe and predict
    future = model.make_future_dataframe(periods=forecast_out)
    forecast = model.predict(future)

    # Return only forecasted values (the "future")
    forecast_df = forecast[['ds', 'yhat']].set_index('ds')
    predicted_series = forecast_df['yhat'].tail(forecast_out)

    return predicted_series

def prophet(start_date, end_date, Num_porSimulation, selected, record_percentage_to_predict):
    frame = {}
    for stock in selected:
        price = price_forecast_prophet(stock, start_date, end_date, record_percentage_to_predict)
        frame[stock] = price
    table = pd.DataFrame(frame)
    pd.DataFrame(frame).to_csv('Out.csv')
    returns_daily = table.pct_change()
    returns_daily.to_csv('Out1.csv')
    returns_annual = ((1 + returns_daily.mean()) ** 254) - 1
    returns_annual.to_csv('Out2.csv')
    cov_daily = returns_daily.cov()
    cov_annual = cov_daily * 250
    port_returns = []
    port_volatility = []
    sharpe_ratio = []
    stock_weights = []
    num_assets = len(selected)
    num_portfolios = Num_porSimulation
    np.random.seed(101)
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        sharpe = returns / volatility
        sharpe_ratio.append(sharpe)
        port_returns.append(returns * 100)
        port_volatility.append(volatility * 100)
        stock_weights.append(weights)
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility,
                 'Sharpe Ratio': sharpe_ratio}
    for counter, symbol in enumerate(selected):
        portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]
    df = pd.DataFrame(portfolio)
    column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock + ' Weight' for stock in selected]
    df = df[column_order]
    min_volatility = df['Volatility'].min()
    max_sharpe = df['Sharpe Ratio'].max()
    max_return = df['Returns'].max()
    max_vol = df['Volatility'].max()
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    min_variance_port = df.loc[df['Volatility'] == min_volatility]
    max_returns = df.loc[df['Returns'] == max_return]
    max_vols = df.loc[df['Volatility'] == max_vol]
    plt.clf()
    plt.style.use('seaborn-v0_8-dark')
    ax = df.plot.scatter(
        x='Volatility', y='Returns', c='Sharpe Ratio',
        cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True, alpha=0.8
    )
    plt.scatter(
        x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'],
        c='green', marker='D', s=200, label='Max Sharpe'
    )
    plt.scatter(
        x=min_variance_port['Volatility'], y=min_variance_port['Returns'],
        c='orange', marker='D', s=200, label='Min Volatility'
    )
    plt.scatter(
        x=max_vols['Volatility'], y=max_returns['Returns'],
        c='red', marker='D', s=200, label='Max Return'
    )
    plt.xlabel('Volatility (Std. Deviation) Percentage %')
    plt.ylabel('Expected Returns Percentage %')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.tight_layout()
    st.subheader("Efficient Frontier")
    with st.expander("What is the Efficient Frontier?"):
        st.markdown("""
    The efficient frontier is a concept from Modern Portfolio Theory. It represents the set of optimal portfolios that offer the highest expected return for a given level of risk (volatility), or the lowest risk for a given level of expected return. 

    - Each point on the frontier is a portfolio that is "efficient"—meaning you can't get a higher return without taking on more risk, or reduce risk without lowering your expected return.
    - Portfolios below the frontier are sub-optimal, because they do not provide enough return for the level of risk.
    - The shape of the frontier helps investors visualize the trade-off between risk and return, and select a portfolio that matches their risk tolerance.

    The chart below shows the simulated portfolios, with the efficient frontier highlighted by the best combinations of risk and return.
    """)
    st.pyplot(plt.gcf())
    red_num = df[df["Returns"] == max_return].index
    yellow_num = df[df['Volatility'] == min_volatility].index
    green_num = df[df['Sharpe Ratio'] == max_sharpe].index
    multseries = pd.Series([1, 1, 1] + [100 for stock in selected],
                        index=['Returns', 'Volatility', 'Sharpe Ratio'] + [stock + ' Weight' for stock in selected])
    optimal_rows = []
    optimal_names = []
    if len(red_num) > 0:
        optimal_rows.append(df.loc[red_num[0]].multiply(multseries))
        optimal_names.append("🚩 Max Returns")
    if len(yellow_num) > 0:
        optimal_rows.append(df.loc[yellow_num[0]].multiply(multseries))
        optimal_names.append("🟨 Min Volatility")
    if len(green_num) > 0:
        optimal_rows.append(df.loc[green_num[0]].multiply(multseries))
        optimal_names.append("🟩 Max Sharpe Ratio")
    if optimal_rows:
        optimal_df = pd.DataFrame(optimal_rows, index=optimal_names)
        st.subheader("Optimal Portfolios Table")
        st.dataframe(optimal_df.style.format("{:.2f}"), hide_index=False)
    else:
        st.info("No optimal portfolios found.")
    with st.expander("Show All Simulated Portfolios DataFrame"):
        st.dataframe(df.style.format("{:.2f}"), hide_index=True)

# Set page configuration for the standalone app
st.set_page_config(
    page_title="Portfolio Optimization Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
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

# Page title and description
st.markdown("<h1 class='main-header'>Modern Portfolio Theory (MPT) Optimizer</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["ℹ️ About MPT", "🔧 How to use?", "📈 Portfolio Optimization"])
# MPT explanation
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown("""
            **📈 What is a Portfolio?**
            
            A portfolio is a collection of investments owned by an individual or organization.
            These investments can include:
            - Stocks (like Apple or Tesla)
            - Bonds
            - ETFs (Exchange-Traded Funds)
            - Real estate
            - Cryptocurrencies
            - Cash or cash equivalents
                        
            **🎯 The Goal of MPT:**

            To maximize expected returns for a given level of risk, or minimize risk for a desired return.

            **🧠 How Does MPT Work?**

            MPT relies on a few key ideas:

            **✅ 1. Diversification**

            Don't put all your eggs in one basket. Holding a mix of investments can reduce your overall risk.
            """)
    with col2:
            st.markdown("""

            **✅ 2. Risk and Return**

            Every investment has:
            - Expected return (how much it might earn)
            - Risk (how much its price might fluctuate)

            MPT finds the best combination that offers the most return for the least risk.

            **✅ 3. Efficient Frontier**

            This is a graph showing the best possible portfolios — ones that offer the highest return for a given level of risk. Investors use this to pick a portfolio that fits their comfort with risk.
            
            **📈 Why Does It Matter?**

            Using MPT, investors can:
            - Build smarter, more balanced portfolios
            - Make decisions based on data, not emotions
            - Improve their long-term financial outcomes
            - 🚀 When combined with tools like machine learning or price forecasting, MPT becomes even more powerful for making forward-looking investment decisions.
            """)

    st.video("https://www.youtube.com/watch?v=YtrMGKLRtwA", start_time=0)

with tab2:
    st.markdown("""
    **🔧 Steps to Optimize Your Portfolio:**
    
    1. **Select Stocks**: Choose the stocks you want to include in your portfolio.
    2. **Set Parameters**: Define the time period and number of simulations for optimization.
    3. **Run Optimization**: Click the button to run the MPT optimization.
    4. **View Results**: Analyze the optimal portfolio, including expected returns, volatility, and weights of each stock.
    
    **📊 Visualization**: The tool will also display a graph of the efficient frontier, helping you visualize the risk-return trade-off of your selected portfolio.
    """)


# Add a session state variable to store the latest optimization results
if 'optimization_result' not in st.session_state:
    st.session_state['optimization_result'] = None
if 'last_params' not in st.session_state:
    st.session_state['last_params'] = {}

with st.sidebar:
    # models
    models = {'Linear Regression': markovich, 'Random Forest': random_forest, 'Prophet': prophet}
    st.subheader("Pick a model for price forecasting")
    selected_model = st.selectbox(
        "Select Model",
        options=list(models.keys()),
        index=0,  # Default to the first model
        help="Choose a model for price forecasting."
    )
    # Select stocks from the UI
    st.subheader("Select Stocks for Your Portfolio")
    # Fetch stock tickers
    tickers = fetch_stock_tickers()
    selected = st.pills(
        "Select Stocks",
        options=tickers,
        selection_mode="multi",
        default=['AAPL', 'MSFT', 'GOOGL'],  # Default selected stocks
        label_visibility="visible",
        help="Select stocks for your portfolio."
    )

    # Add options to pick start_date, end_date, Num_porSimulation, record_percentage_to_predict
    st.subheader("Portfolio Optimization Parameters")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.datetime.now())

    Num_porSimulation = st.slider(
        "Number of Portfolio Simulations",
        min_value=1000,
        max_value=10000,
        value=5000,
        step=1000,
        help="Number of random portfolios to simulate."
    )

    record_percentage_to_predict = st.slider(
        "Percentage of Data to Use for Prediction",
        min_value=0.05,
        max_value=0.30,
        value=0.20,
        step=0.01,
        help="Fraction of data to use for forecasting (e.g., 0.2 = 20%)."
    )

    run_opt = st.button("Run Optimization", type="primary")

# Prepare parameters as a dict for comparison
current_params = {
    "start_date": start_date,
    "end_date": end_date,
    "Num_porSimulation": Num_porSimulation,
    "selected": tuple(selected),
    "record_percentage_to_predict": record_percentage_to_predict,
}

def run_and_store_optimization():
    # Clear previous matplotlib figures to avoid memory leaks
    plt.close('all')
    # Run optimization and store a function to re-render results
    def render():
        models[selected_model](start_date, end_date, Num_porSimulation, selected, record_percentage_to_predict)
    
    st.session_state['optimization_result'] = render
    st.session_state['last_params'] = current_params

# Remove the cache_resource function and use direct calls instead
with tab3:
    # Use a static variable to track if optimization has been run
    if "run_opt" not in st.session_state:
        st.session_state["run_opt"] = False
        st.session_state["run_opt_params"] = None

    if run_opt:
        st.session_state["run_opt"] = True
        st.session_state["run_opt_params"] = {
            "start_date": start_date,
            "end_date": end_date,
            "Num_porSimulation": Num_porSimulation,
            "selected": tuple(selected),
            "record_percentage_to_predict": record_percentage_to_predict,
        }
        st.rerun()

    if st.session_state.get("run_opt", False):
        params = st.session_state["run_opt_params"]
        # Directly call the selected model function instead of cached_model
        models[selected_model](
            params["start_date"],
            params["end_date"],
            params["Num_porSimulation"],
            list(params["selected"]),
            params["record_percentage_to_predict"]
        )
    else:
        st.info("Click the button to run the portfolio optimization.")
        st.markdown("""
        **ℹ️ Note**: The optimization process may take some time depending on the number of simulations and selected stocks.
        """)