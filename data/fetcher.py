import yfinance as yf
import pandas as pd

def fetch_history(tickers: list, start: str, end: str) -> dict:
    """
    Descarga historial de OHLCV para una lista de tickers.
    Retorna un diccionario {ticker: DataFrame}.
    """
    if not tickers:
        return {}
    
    if len(tickers) == 1:
        df = yf.download(tickers[0], start=start, end=end, progress=False)
        if df.empty:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return {tickers[0]: df}
    
    data = yf.download(tickers, start=start, end=end, group_by="ticker", progress=False)
    
    result = {}
    for ticker in tickers:
        try:
            df = data[ticker].dropna(how='all')
            if not df.empty:
                result[ticker] = df
        except KeyError:
            continue
            
    return result
