import sys, os, datetime
sys.path.insert(0, os.path.abspath('.'))
from web.app_trading import analyze_ticker_for_today

end = datetime.date.today().strftime('%Y-%m-%d')
start = (datetime.date.today() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')

print("Analyzing NVDA...")
try:
    res = analyze_ticker_for_today("NVDA", "CALM", 4.2, start, end)
    print(f"Result: {res}")
except Exception as e:
    import traceback
    traceback.print_exc()
