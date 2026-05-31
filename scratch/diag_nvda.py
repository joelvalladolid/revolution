import sys, os, datetime
sys.path.insert(0, os.path.abspath('.'))
from web.app_trading import get_ticker_technicals

end = datetime.date.today().strftime('%Y-%m-%d')
start = (datetime.date.today() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')

print("Fetching NVDA...")
try:
    df, ind, price = get_ticker_technicals("NVDA", start, end)
    print(f"Result: df is {type(df)}")
except Exception as e:
    import traceback
    traceback.print_exc()
