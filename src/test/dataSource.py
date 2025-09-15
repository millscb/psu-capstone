import yfinance as yf

data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
print(data.head())