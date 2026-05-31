import pandas as pd

def load_top(file_path):
    df = pd.read_csv(file_path)
    # Filter 5 days
    df = df[df['horizon_days'] == 5].copy()
    df['name'] = df['indicator'] + " <=" + df['threshold'].astype(str)
    # EMA and Volume and MACD and OBV are "above" but let's just use threshold
    df.loc[df['indicator'].str.contains('EMA|Volume|MACD|OBV'), 'name'] = df['indicator'] + " >=" + df['threshold'].astype(str)
    return df[['name', 'edge', 'n_signals', 'indicator', 'threshold']].set_index('name')

try:
    df_full = load_top('tournament_results_full.csv')
    df_bear = load_top('tournament_results_bear_2022.csv')
    df_covid = load_top('tournament_results_covid.csv')
    
    # Get top 15 from full period
    top15_full = df_full.sort_values(by='edge', ascending=False).head(15)
    
    print(f"{'Indicador':<20} | {'Edge FULL':<10} | {'Edge BEAR':<10} | {'Edge COVID':<10} | {'Señales FULL':<12}")
    print("-" * 75)
    
    for name, row in top15_full.iterrows():
        edge_full = row['edge'] * 100
        n_full = row['n_signals']
        
        edge_bear = df_bear.loc[name, 'edge'] * 100 if name in df_bear.index else float('nan')
        edge_covid = df_covid.loc[name, 'edge'] * 100 if name in df_covid.index else float('nan')
        
        print(f"{name:<20} | {edge_full:>8.2f} pp | {edge_bear:>8.2f} pp | {edge_covid:>8.2f} pp | {n_full:>10.0f}")
        
except Exception as e:
    print(f"Error: {e}")
