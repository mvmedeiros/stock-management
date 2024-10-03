import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

#******* BASIC DATA PROCESSING
# Load transactions data
df_transactions = pd.read_csv('transactions_mocked.csv', 
                              usecols=['Date',
                                       'Transaction',
                                       'Ticker',
                                       'Shares',
                                       'Price',
                                       'Transaction Value',
                                       'Settlement Fees',
                                       'Fees']
                              )

# Stock historical data on close price
df_stock_close = pd.read_csv('stock_history_close.csv')
df_stock_close['Date'] = pd.to_datetime(df_stock_close['Date'])

# *************** PORTIFOLIO PROCESSING **************
# Creates a portifolio dataframe considering the transactions dataframe as base for further processing and expansion
df_portifolio = df_transactions

# Convert "Data" column to date format
df_portifolio['Date'] = pd.to_datetime(df_portifolio['Date'])
df_portifolio['Month'] = df_portifolio['Date'].apply(lambda each: str(each.month) + '/' + str(each.year))

# Depending on purchase or selling operation, defines if the stock shares are an increase or decrease
df_portifolio['Shares'] = df_transactions.apply(
    lambda row:
    row['Shares'] if row['Transaction'] == 'BUY' else row['Shares']*-1, axis=1
    )

# Inicializar colunas de resultado
df_portifolio['Total Shares'] = 0
df_portifolio['Position Purchase Cost'] = 0
df_portifolio['Average Price'] = 0.0

# Iteration for each ticker on dataframe --> FUTURE REFACTORING TO VECTORIZED OPERATIONS DUE TO EXECUTION PERFORMANCE ISSUES FOR HUGE DATA
for ticker in df_portifolio['Ticker'].unique():
    ticker_mask = df_portifolio['Ticker'] == ticker
    cumulative_shares = 0
    cumulative_purchase_cost = 0.0
    
    # Iterar sobre as operações de cada ticker
    for idx, row in df_portifolio[ticker_mask].iterrows():
        if row['Shares'] > 0:  # Updates total shares and costs for purchases
            cumulative_shares += row['Shares']
            cumulative_purchase_cost += row['Transaction Value'] + row['Settlement Fees'] + row['Fees']
        elif row['Shares'] < 0:  # Venda
            # Reduce total shares positions on sellings
            shares_sold = -row['Shares'] # Set value to positive to simplify calculations
            if cumulative_shares >= shares_sold:
                # Update total shares and costs proportionally
                avg_cost_per_share = cumulative_purchase_cost / cumulative_shares
                cumulative_purchase_cost -= avg_cost_per_share * shares_sold
                cumulative_shares -= shares_sold
            else:
                # Once the position is liquidated the total shares and purchase costs are set to zero
                cumulative_shares = 0
                cumulative_purchase_cost = 0.0

        # Updating average price
        avg_price = cumulative_purchase_cost / cumulative_shares if cumulative_shares > 0 else 0

        # Updates the dataframe with the correspondent values
        df_portifolio.loc[idx, 'Total Shares'] = cumulative_shares
        df_portifolio.loc[idx, 'Position Purchase Cost'] = cumulative_purchase_cost
        df_portifolio.loc[idx, 'Average Price'] = avg_price

# Profit calculation
df_portifolio['Selling Profit'] = df_portifolio.apply(
    lambda row: ((row['Transaction Value'] - row['Settlement Fees'] - row['Fees']) - (row['Average Price']*-row['Total Shares'])) if row['Shares'] < 0 else 0, axis=1
)

# Expand transactions to every market trading day
# Melting historic data
df_stock_close_melted = df_stock_close.melt(id_vars='Date', var_name='Ticker', value_name='Close Price')

# Step 1: Create a complete date range from 'stocks'
complete_dates = df_stock_close_melted['Date'].unique()

# Step 2: Create a MultiIndex with all tickers and the complete date range
tickers = df_portifolio['Ticker'].unique()
multi_index = pd.MultiIndex.from_product([complete_dates, tickers], names=['Date', 'Ticker'])

# Step 3: Create a new DataFrame with the MultiIndex
expanded_transactions = pd.DataFrame(index=multi_index)

# Step 4: Reset index of the original transactions DataFrame for merging
df_portifolio['Date'] = pd.to_datetime(df_portifolio['Date'])

df_portifolio = pd.merge(expanded_transactions, df_portifolio, how='left', left_index=True, right_on=['Date', 'Ticker'])
df_portifolio = df_portifolio.merge(df_stock_close_melted, on=['Date', 'Ticker'], how='left', suffixes=('', '_left'))

df_portifolio[['Total Shares','Position Purchase Cost', 'Average Price']] = df_portifolio.groupby('Ticker')[['Total Shares','Position Purchase Cost', 'Average Price']].ffill()
df_portifolio['Transaction'] = df_portifolio.apply(lambda row: 
                                     'CARRY' if (pd.isna(row['Shares']) and row['Total Shares'] > 0) 
                                     else row['Transaction'], axis=1)
df_portifolio = df_portifolio.dropna(subset=['Transaction'])

# ******* Position Market Value CALCULATION
df_portifolio['Total Equity'] = df_portifolio.apply(
    lambda row: row['Close Price'] * row['Total Shares'], axis=1
)

# Equity variation on time
df_portifolio['Variation'] = df_portifolio.apply(
    lambda row: row['Total Equity'] - row['Position Purchase Cost'], axis=1
)

# Percentual change
df_portifolio['Rentability'] = df_portifolio.groupby('Ticker')['Total Equity'].pct_change()

df_portifolio['Rentability'] = df_portifolio.groupby('Ticker')['Rentability'].transform(
    lambda row: (1 + row).cumprod() - 1)

# Format numeric informations
df_portifolio['Average Price'] = df_portifolio['Average Price'].round(2)

# The total equity will be calculated grouping each Date from portifolio dataframe
# to enable summing each date total equity. As several transactions for a same ticker
# can be recorded, the strategy must consider only the last from each group of Date  and Ticker
# on portifolio dataframe
df_total_equity = df_portifolio[['Date', 'Ticker', 'Total Equity']].groupby(['Date','Ticker']).tail(1)
df_total_equity = df_total_equity.groupby('Date')['Total Equity'].sum().reset_index()
df_total_equity['Ticker'] = 'Total Equity'

# ***** CURRENT PORTIFOLIO *********
# To get the latest portifolio status it's necessary to get the last occurrence of every
# stock in the database with a number of total share above zero

df_portifolio_current = df_portifolio.groupby('Ticker').tail(1)
df_portifolio_current = df_portifolio_current.loc[df_portifolio_current['Total Shares'] > 0].reset_index(drop=True)

total_equity = df_portifolio_current['Total Equity'].sum()
total_variation = df_portifolio_current['Variation'].sum()
last_update_date = df_portifolio_current['Date'].iloc[-1]

#******* DASHBOARD BUILDING
# Configure page layout
st.set_page_config(layout='wide')

# Definir as opções de páginas
pages = ["Equity Analysis", 'Transactions', 'Rentability', "Dividends"]

# Criar uma seleção de páginas na barra lateral
page_selected = st.sidebar.radio("Select the Dashboard Page:", pages)

# Condicional para exibir o conteúdo baseado na página selecionada
if page_selected == "Equity Analysis":
    # ***** TOTAL EQUITY GRAPH *********

    # Selection box for Tickers visualization
    selected_tickers = st.multiselect("Select the Tickers to display", tickers, default=tickers)

    # Filter portifolio dataframe for selected tickers
    df_filtered_equity = df_portifolio[df_portifolio['Ticker'].isin(selected_tickers)]

    # Create a Plotly Express line chart for equity
    fig_total_equity = px.area(df_total_equity, x='Date',color='Ticker',
                    y='Total Equity', 
                    title='Total Equity Over Time', labels={'Total Equity': 'Total Equity'})


    fig_tickers_equity = px.area(df_filtered_equity, x='Date',color='Ticker',
                        y='Total Equity', 
                        title='Total Equity Over Time', labels={'Total Equity': 'Total Equity'})
    

    # Create a subplot with 1 row and 1 column
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Create the first area chart using Plotly Express for the first ticker
    fig1 = px.line(df_total_equity, x='Date', y='Total Equity', color='Ticker',title='Total Equity', labels={'Total Equity': 'Total Equity'})
    # Add the first area chart to the subplot
    for trace in fig1.data:
        fig.add_trace(trace, secondary_y=False)

    # Create the second area chart using Plotly Express for different tickers
    fig2 = px.area(df_filtered_equity, x='Date', y='Total Equity', color='Ticker', labels={'Total Equity': 'Equity'})
    # Add the second area chart to the subplot
    for trace in fig2.data:
        fig.add_trace(trace, secondary_y=True)

    # Update layout
    fig.update_layout(
        title_text="Total Equity and Stock Equity Over Time",
        xaxis_title="Date",
        yaxis_title="Brazilian Reals (R$)",
    )

    # Set the Y-axis range for both axes
    y_axis_range = [0, df_total_equity['Total Equity'].max()] 
    fig.update_yaxes(range=y_axis_range, secondary_y=False)
    fig.update_yaxes(range=y_axis_range, showgrid=False, showticklabels=False, secondary_y=True)

    with st.container(border=True):
        st.plotly_chart(fig)

    # ******** PORTIFOLIO CURRENT STATUS
    # Labels for portifolio pie chart
    columns_to_use = ['Ticker', 'Total Shares', 'Average Price', 'Position Purchase Cost', 'Close Price', 'Total Equity', 'Variation', 'Rentability']
    portifolio_current_stocks = df_portifolio_current['Ticker'].unique()
    fig_portifolio_chart = px.pie(df_portifolio_current, values=df_portifolio_current['Total Equity'], names=df_portifolio_current['Ticker'])

    # STOCK PORTIFOLIO CONTAINER
    with st.container(border=True):
        st.markdown(f"Stocks Portifolio at {last_update_date:%Y-%m-%d}")
        st.markdown(f"TOTAL EQUITY: {total_equity :,.2f}")
        st.markdown(f"TOTAL VARIATION: {total_variation:,.2f}")
        # Create two columns for portifolio table and graphical visualization
        col1, col2 = st.columns(2)

        # Column 1 content
        with col1:
            st.dataframe(df_portifolio_current[columns_to_use], use_container_width=True, height=500)

        # Column 2 content
        with col2:
            st.write(fig_portifolio_chart)
        
        fig = px.line(df_portifolio, x='Date', y='Rentability', color='Ticker')
        st.write(fig)

elif page_selected == "Transactions":
    st.markdown("Under development")

elif page_selected == "Rentability":
    st.markdown("Under development")


elif page_selected == "Dividends":
    st.markdown("Under development")