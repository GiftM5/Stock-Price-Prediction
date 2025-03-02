#handlesatock data
import pandas as pd
#plot stock price trends
import matplotlib.pyplot as plt
#fetching data from yahoo finance
import yfinance as yf

stock_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-01-01'
print('Fetching data for stock:', stock_symbol)
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
print(stock_data.head())
print(stock_data.tail())

stock_data.to_csv('stock_data.csv')
print('Data saved to stock_data.csv')

plt.figure(figsize=(12, 5))
plt.plot(stock_data["Close"], label="Closing Price")
plt.title(f"{stock_symbol} Stock Closing Price")
plt.xlabel("Date")  
plt.ylabel("Stock Price (USD)")  
plt.legend()
plt.show()




