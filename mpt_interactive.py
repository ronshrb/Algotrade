import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize

def get_stock_data(tickers, start_date, end_date):
    """
    Fetch stock data for the specified tickers within the date range.
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: DataFrame with stock price data
    """
    data = {}
    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            data[ticker] = stock_data['Adj Close']
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            
    return pd.DataFrame(data)

def calculate_portfolio_performance(weights, returns, cov_matrix):
    """
    Calculate portfolio performance metrics.
    
    Args:
        weights (np.array): Portfolio weights
        returns (np.array): Expected returns
        cov_matrix (np.array): Covariance matrix
        
    Returns:
        tuple: (Portfolio return, Portfolio volatility, Sharpe ratio)
    """
    portfolio_return = np.sum(returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

def generate_random_portfolios(num_portfolios, returns, cov_matrix, num_assets):
    """
    Generate random portfolios for visualization.
    
    Args:
        num_portfolios (int): Number of portfolios to generate
        returns (np.array): Expected returns
        cov_matrix (np.array): Covariance matrix
        num_assets (int): Number of assets
        
    Returns:
        tuple: (Returns, Volatilities, Sharpe ratios, Weights)
    """
    results = np.zeros((3, num_portfolios))
    weights_record = np.zeros((num_portfolios, num_assets))
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_performance(weights, returns, cov_matrix)
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
        
        weights_record[i, :] = weights
        
    return results[0], results[1], results[2], weights_record

def optimize_portfolio(objective, returns, cov_matrix, num_assets, target_return=None):
    """
    Optimize portfolio based on the specified objective.
    
    Args:
        objective (str): 'min_volatility', 'max_sharpe', or 'efficient_frontier'
        returns (np.array): Expected returns
        cov_matrix (np.array): Covariance matrix
        num_assets (int): Number of assets
        target_return (float, optional): Target return for efficient frontier
        
    Returns:
        np.array: Optimized weights
    """
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array([1/num_assets] * num_assets)
    
    if objective == 'min_volatility':
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        result = minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
    elif objective == 'max_sharpe':
        def negative_sharpe_ratio(weights):
            portfolio_return = np.sum(returns * weights) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return -portfolio_return / portfolio_volatility
        
        result = minimize(negative_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
    elif objective == 'efficient_frontier':
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(returns * x) * 252 - target_return}
        ]
        
        result = minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
    return result['x']

def display_mpt_page():
    """
    Display the Modern Portfolio Theory interactive page.
    """
    # Page title and description
    st.markdown("<h1 class='main-header'>Modern Portfolio Theory (MPT) Optimizer</h1>", unsafe_allow_html=True)
    
    # MPT explanation
    with st.expander("🔍 What is Modern Portfolio Theory?"):
        st.markdown("""
        ### Modern Portfolio Theory (MPT)
        
        Modern Portfolio Theory (MPT) is an investment framework developed by Harry Markowitz in 1952. It was revolutionary because it was the first mathematical framework that quantified the relationship between risk and return in a portfolio.
        
        **Key Concepts:**
        
        1. **Risk and Return Relationship**: MPT establishes that risk and return are directly related, but through diversification, investors can optimize this relationship.
        
        2. **Diversification**: By combining assets that are not perfectly correlated, investors can reduce portfolio risk without necessarily sacrificing returns.
        
        3. **Efficient Frontier**: This is the set of optimal portfolios that offer the highest expected return for a given level of risk, or the lowest risk for a given level of return.
        
        4. **Risk Measures**: 
           - **Volatility** (Standard Deviation): Measures how much returns fluctuate around the average.
           - **Sharpe Ratio**: Risk-adjusted return, calculated as (expected return - risk-free rate) / standard deviation.
        
        **Goal of MPT:**
        The goal is to create an "efficient portfolio" that maximizes returns for a given level of risk, or minimizes risk for a given level of return.
        
        **Using the MPT Optimizer Tool:**
        1. Select stocks/assets for your portfolio
        2. Choose the date range for historical data
        3. The tool will generate:
           - The efficient frontier
           - The optimal portfolio for maximizing returns
           - The optimal portfolio for minimizing risk (minimum variance)
           - The optimal portfolio for maximizing risk-adjusted returns (Sharpe ratio)
        """)

    # Input parameters    
    st.markdown("<h2 class='sub-header'>Configure Your Portfolio</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Stock selection
        default_tickers = "SPY,QQQ,GLD,TLT,VGK"
        stock_input = st.text_input(
            "Enter stock symbols separated by commas (e.g., AAPL,MSFT,GOOG):",
            default_tickers
        )
        selected_tickers = [ticker.strip() for ticker in stock_input.split(',')]
        
    with col2:
        # Date range selection
        today = datetime.today()
        default_start = today - timedelta(days=5*365)  # 5 years ago
        start_date = st.date_input("Start Date:", default_start)
        end_date = st.date_input("End Date:", today)
        
    # Number of simulations
    num_portfolios = st.slider("Number of portfolio simulations:", min_value=1000, max_value=10000, value=3000, step=500)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        risk_free_rate = st.slider("Risk-free rate (%):", min_value=0.0, max_value=5.0, value=2.0, step=0.1) / 100
        show_individual_stocks = st.checkbox("Show individual stocks on the chart", value=True)
        plot_type = st.radio("Plot Type:", ["Interactive (Plotly)", "Static (Matplotlib)"])
    
    if st.button("Generate Portfolio Analysis"):
        with st.spinner("Fetching data and calculating optimal portfolios..."):
            try:
                # Fetch data
                price_data = get_stock_data(selected_tickers, start_date, end_date)
                
                if price_data.empty or price_data.shape[1] < 2:
                    st.error("Not enough stock data available. Please select different stocks or date range.")
                    return
                
                # Calculate returns and covariance
                returns = price_data.pct_change().dropna()
                mean_returns = returns.mean()
                cov_matrix = returns.cov()
                
                # Calculate annualized returns and covariance
                ann_returns = (1 + mean_returns) ** 252 - 1
                ann_cov = cov_matrix * 252
                
                # Generate random portfolios
                returns_arr = ann_returns.values
                cov_arr = ann_cov.values
                num_assets = len(selected_tickers)
                
                port_returns, port_volatility, port_sharpe, weights_record = generate_random_portfolios(
                    num_portfolios, returns_arr, cov_arr, num_assets
                )
                
                # Find the optimal portfolios
                min_vol_weights = optimize_portfolio('min_volatility', returns_arr, cov_arr, num_assets)
                max_sharpe_weights = optimize_portfolio('max_sharpe', returns_arr, cov_arr, num_assets)
                
                min_vol_return, min_vol_volatility, min_vol_sharpe = calculate_portfolio_performance(
                    min_vol_weights, returns_arr, cov_arr
                )
                
                max_sharpe_return, max_sharpe_volatility, max_sharpe_sharpe = calculate_portfolio_performance(
                    max_sharpe_weights, returns_arr, cov_arr
                )
                
                # Create portfolio results dataframe
                portfolio_results = {
                    'Returns': port_returns,
                    'Volatility': port_volatility,
                    'Sharpe Ratio': port_sharpe
                }
                
                for i, symbol in enumerate(selected_tickers):
                    portfolio_results[f'{symbol} Weight'] = weights_record[:, i]
                    
                results_df = pd.DataFrame(portfolio_results)
                
                # Visualize the efficient frontier
                st.markdown("<h2 class='sub-header'>Efficient Frontier</h2>", unsafe_allow_html=True)
                
                if plot_type == "Interactive (Plotly)":
                    # Create plotly scatter plot
                    fig = px.scatter(
                        results_df, 
                        x='Volatility', 
                        y='Returns',
                        color='Sharpe Ratio',
                        color_continuous_scale='RdYlGn',
                        title='Portfolio Optimization: Efficient Frontier',
                        hover_data=[f'{ticker} Weight' for ticker in selected_tickers]
                    )
                    
                    # Add optimal portfolios
                    fig.add_trace(
                        go.Scatter(
                            x=[min_vol_volatility],
                            y=[min_vol_return],
                            mode='markers',
                            marker=dict(color='yellow', size=15, symbol='diamond'),
                            name='Minimum Volatility Portfolio'
                        )
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[max_sharpe_volatility],
                            y=[max_sharpe_return],
                            mode='markers',
                            marker=dict(color='green', size=15, symbol='diamond'),
                            name='Maximum Sharpe Ratio Portfolio'
                        )
                    )
                    
                    # Add individual stocks if requested
                    if show_individual_stocks:
                        # Calculate individual stock returns and volatilities
                        for i, ticker in enumerate(selected_tickers):
                            stock_return = ann_returns[i] 
                            stock_volatility = np.sqrt(ann_cov[i, i])
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=[stock_volatility],
                                    y=[stock_return],
                                    mode='markers+text',
                                    marker=dict(color='blue', size=10),
                                    text=ticker,
                                    textposition="top center",
                                    name=ticker
                                )
                            )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title='Annualized Volatility (Standard Deviation)',
                        yaxis_title='Annualized Return',
                        xaxis=dict(tickformat=".2%"),
                        yaxis=dict(tickformat=".2%"),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:  # Static Matplotlib plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    scatter = ax.scatter(
                        port_volatility, 
                        port_returns, 
                        c=port_sharpe,
                        cmap='RdYlGn',
                        alpha=0.7
                    )
                    
                    plt.colorbar(scatter, label='Sharpe Ratio')
                    
                    # Plot the optimal portfolios
                    ax.scatter(
                        min_vol_volatility, 
                        min_vol_return, 
                        marker='D', 
                        color='yellow', 
                        s=100, 
                        label='Minimum Volatility'
                    )
                    
                    ax.scatter(
                        max_sharpe_volatility, 
                        max_sharpe_return, 
                        marker='D', 
                        color='green', 
                        s=100, 
                        label='Maximum Sharpe Ratio'
                    )
                    
                    # Add individual stocks if requested
                    if show_individual_stocks:
                        for i, ticker in enumerate(selected_tickers):
                            stock_return = ann_returns[i] 
                            stock_volatility = np.sqrt(ann_cov[i, i])
                            ax.scatter(
                                stock_volatility, 
                                stock_return, 
                                marker='o', 
                                color='blue', 
                                s=50
                            )
                            ax.annotate(
                                ticker, 
                                (stock_volatility, stock_return),
                                xytext=(5, 5),
                                textcoords='offset points'
                            )
                    
                    plt.title('Portfolio Optimization: Efficient Frontier')
                    plt.xlabel('Annualized Volatility')
                    plt.ylabel('Annualized Return')
                    plt.legend()
                    plt.grid(True)
                    
                    st.pyplot(fig)
                
                # Display optimal portfolio details
                st.markdown("<h2 class='sub-header'>Optimal Portfolios</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Minimum Volatility Portfolio")
                    st.markdown(f"**Expected Annual Return:** {min_vol_return:.2%}")
                    st.markdown(f"**Expected Annual Volatility:** {min_vol_volatility:.2%}")
                    st.markdown(f"**Sharpe Ratio:** {min_vol_sharpe:.2f}")
                    
                    # Create a dataframe for the weights
                    min_vol_weights_df = pd.DataFrame({
                        'Asset': selected_tickers,
                        'Weight': min_vol_weights
                    })
                    
                    # Create a pie chart for weights
                    fig = px.pie(
                        min_vol_weights_df,
                        values='Weight',
                        names='Asset',
                        title='Asset Allocation'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### Maximum Sharpe Ratio Portfolio")
                    st.markdown(f"**Expected Annual Return:** {max_sharpe_return:.2%}")
                    st.markdown(f"**Expected Annual Volatility:** {max_sharpe_volatility:.2%}")
                    st.markdown(f"**Sharpe Ratio:** {max_sharpe_sharpe:.2f}")
                    
                    # Create a dataframe for the weights
                    max_sharpe_weights_df = pd.DataFrame({
                        'Asset': selected_tickers,
                        'Weight': max_sharpe_weights
                    })
                    
                    # Create a pie chart for weights
                    fig = px.pie(
                        max_sharpe_weights_df,
                        values='Weight',
                        names='Asset',
                        title='Asset Allocation'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Allow users to download the results
                st.markdown("<h2 class='sub-header'>Download Results</h2>", unsafe_allow_html=True)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Portfolio Simulations CSV",
                    data=csv,
                    file_name='portfolio_simulations.csv',
                    mime='text/csv',
                )
                
                # Display correlation matrix
                st.markdown("<h2 class='sub-header'>Asset Correlation Matrix</h2>", unsafe_allow_html=True)
                correlation_matrix = returns.corr()
                
                # Plotly heatmap
                fig = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title='Asset Correlation Matrix'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check your inputs and try again.")

if __name__ == '__main__':
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
    
    # Display the app
    display_mpt_page()
    
    # Add footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 10px; color: #777;">
        <p>Modern Portfolio Theory Optimization Tool</p>
        <p>Developed for AlgoTrade Project</p>
    </div>
    """, unsafe_allow_html=True)
