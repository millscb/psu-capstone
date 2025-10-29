import sys
import requests
import yfinance as yf


def fetch(ticker: str, start: str, end: str):
    # Let yfinance manage its own HTTP session (curl_cffi); just disable threads/progress
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        threads=False,
        auto_adjust=False,
        group_by="column",
    )
    if df.empty:  # type: ignore
        raise RuntimeError(
            f"No data returned for {ticker}. Check network/proxy or try again."
        )
    return df


if __name__ == "__main__":
    ticker = "AAPL"
    start = "2020-01-01"
    end = "2021-01-01"
    try:
        df = fetch(ticker, start, end)
        print(df.head())  # type: ignore
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        # Quick connectivity probe (optional)
        try:
            r = requests.get(
                "https://query2.finance.yahoo.com/v10/finance/quoteSummary/AAPL?modules=price",
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            print(f"Yahoo API status: {r.status_code}")
        except Exception as p:
            print(f"Probe failed: {p}", file=sys.stderr)
        sys.exit(1)
