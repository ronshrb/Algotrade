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

def fetch_stock_tickers():
    """
    Fetches stock tickers from major indices and returns them as a DataFrame
    """
    # Define major indices to fetch tickers from
    indices = {
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ': '^IXIC',
        'Russell 2000': '^RUT'
    }
    
    all_tickers = []
    
    try:
        # S&P 500 companies
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500_tickers = sp500.set_index('Symbol')['Security'].to_dict()
        for ticker, name in sp500_tickers.items():
            all_tickers.append({
                'Ticker': ticker,
                'Company': name,
                'Index': 'S&P 500'
            })
        
        # Add some popular tech stocks that might not be in the indices
        popular_tickers = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com, Inc.',
            'META': 'Meta Platforms, Inc.',
            'NFLX': 'Netflix, Inc.',
            'TSLA': 'Tesla, Inc.',
            'NVDA': 'NVIDIA Corporation',
            'AMD': 'Advanced Micro Devices, Inc.',
            'INTC': 'Intel Corporation'
        }
        
        for ticker, name in popular_tickers.items():
            if not any(t['Ticker'] == ticker for t in all_tickers):
                all_tickers.append({
                    'Ticker': ticker,
                    'Company': name,
                    'Index': 'Popular'
                })
                
    except Exception as e:
        st.warning(f"Error fetching stock data: {e}")
        # Fallback to basic tickers if web scraping fails
        fallback_tickers = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com, Inc.',
            'FB': 'Meta Platforms, Inc.',
            'NFLX': 'Netflix, Inc.',
            'TSLA': 'Tesla, Inc.',
            'NVDA': 'NVIDIA Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'V': 'Visa Inc.'
        }
        for ticker, name in fallback_tickers.items():
            all_tickers.append({
                'Ticker': ticker,
                'Company': name,
                'Index': 'Popular'
            })
    
    return pd.DataFrame(all_tickers)

def fetch_stock_tickers():
    """
    Fetches stock tickers from major indices and returns them as a DataFrame
    """
    # # Define major indices to fetch tickers from
    # indices = {
    #     'S&P 500': '^GSPC',
    #     'Dow Jones': '^DJI',
    #     'NASDAQ': '^IXIC',
    #     'Russell 2000': '^RUT'
    # }
    
    all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'TSLA', 'NVDA', 'AMD', 'INTC']
    
    # try:
    #     # S&P 500 companies
    #     sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    #     sp500_tickers = sp500.set_index('Symbol')['Security'].to_dict()
    #     for ticker, name in sp500_tickers.items():
    #         all_tickers.append(ticker)

                
    # except Exception as e:
    #     st.warning(f"Error fetching stock data: {e}")
    #     # Fallback to basic tickers if web scraping fails
    #     all_tickers = ['APPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'TSLA', 'NVDA', 'AMD', 'INTC']
    
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
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(confidence)
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

    def format_portfolio(portfolio_row, color, title):
        st.markdown(
            f"<div style='background-color:{color};padding:1em;border-radius:0.5em;margin-bottom:1em;'>"
            f"<b>{title}</b><br><pre style='font-size:1em'>{portfolio_row.multiply(multseries).to_string()}</pre>"
            "</div>",
            unsafe_allow_html=True
        )

    st.subheader("Optimal Portfolios")
    if len(red_num) > 0:
        format_portfolio(df.loc[red_num[0]], "#ffebee", "🚩 Max Returns Portfolio")
    if len(yellow_num) > 0:
        format_portfolio(df.loc[yellow_num[0]], "#fffde7", "🟨 Minimum Volatility Portfolio")
    if len(green_num) > 0:
        format_portfolio(df.loc[green_num[0]], "#e8f5e9", "🟩 Maximum Sharpe Ratio Portfolio")

    # Optionally, show the dataframe
    with st.expander("Show All Simulated Portfolios DataFrame"):
        st.dataframe(df.style.format("{:.2f}"), hide_index=True)


# Set page configuration for the standalone app
st.set_page_config(
    page_title="Portfolio Optimization Tool",
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


with st.sidebar:
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

with tab3:
    if run_opt:
        markovich(start_date, end_date, Num_porSimulation, selected, record_percentage_to_predict)
    else:
        st.info("Click the button to run the portfolio optimization.")
        st.markdown("""
        **ℹ️ Note**: The optimization process may take some time depending on the number of simulations and selected stocks.
        """)