import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.data_loader import download_price_data
from src.optimizer import calculate_optimal_portfolio, get_discrete_allocation
from src.simulator import run_monte_carlo_simulation
from src.utils import format_weights

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📈 Portfolio Optimizer", layout="wide")

st.title("💼 Portfolio Optimization App")
st.markdown("Optimize your portfolio using **Modern Portfolio Theory** and **Monte Carlo Simulations**.")

# Sidebar – Inputs
with st.sidebar:
    st.header("🔧 Configuration")
    tickers = st.text_input("Enter tickers (comma separated)", "AAPL,MSFT,GOOGL,TSLA")
    capital = st.number_input("Investment amount ($)", value=10000, step=500)
    optimizer_choice = st.radio("Optimization method", ["Max Sharpe Ratio", "Monte Carlo Simulation"])
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0) / 100
    simulate_n = st.slider("Simulations (for Monte Carlo)", 1000, 20000, 5000, step=1000)

# Process tickers
ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

if st.button("🚀 Run Optimization") and ticker_list:
    with st.spinner("Fetching data and optimizing..."):
        prices = download_price_data(ticker_list, start_date="2020-01-01")
        
        if prices.empty:
            st.error("Could not load data. Check ticker symbols.")
        else:
            st.success("Data loaded.")

            if optimizer_choice == "Max Sharpe Ratio":
                weights, ret, vol, sharpe = calculate_optimal_portfolio(prices, risk_free_rate)
                alloc, leftover = get_discrete_allocation(prices, weights, capital)

                st.subheader("📊 Optimal Portfolio (Sharpe Ratio Maximized)")
                st.dataframe(format_weights(weights))

                st.metric("Expected Return", f"{ret:.2%}")
                st.metric("Volatility", f"{vol:.2%}")
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                st.markdown(f"💵 **Allocation of ${capital:,.2f}:**")
                st.json(alloc)
                st.caption(f"Unallocated Cash: ${leftover:.2f}")

            else:  # Monte Carlo
                df, max_sharpe, min_vol = run_monte_carlo_simulation(prices, num_portfolios=simulate_n, risk_free_rate=risk_free_rate)

                st.subheader("📈 Monte Carlo Simulation Results")
                st.markdown(f"{simulate_n} portfolios simulated. Below: Risk vs Return.")

                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x="Volatility", y="Return", hue="SharpeRatio", palette="viridis", alpha=0.8, ax=ax)
                ax.scatter(max_sharpe["Volatility"], max_sharpe["Return"], color='red', marker='*', s=200, label='Max Sharpe')
                ax.scatter(min_vol["Volatility"], min_vol["Return"], color='blue', marker='*', s=200, label='Min Volatility')
                ax.legend()
                st.pyplot(fig)

                st.subheader("🥇 Best Sharpe Portfolio")
                st.dataframe(format_weights({k: v for k, v in max_sharpe.items() if k in ticker_list}))
                st.metric("Sharpe Ratio", f"{max_sharpe['SharpeRatio']:.2f}")

else:
    st.info("Enter tickers and click 'Run Optimization' to begin.")
