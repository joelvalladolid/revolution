import pandas as pd

def load_csv(path):
    df = pd.read_csv(path)
    df = df[df['horizon_days'] == 5]
    return df.set_index('indicator')

df_single = load_csv('tournament_results_full_single.csv')
df_combo = load_csv('tournament_results_full_combos.csv')

print("1. TOP 5 COMBOS (Por Edge FULL)\n")
print(f"{'Indicador Combo':<45} | {'Edge CALM':<10} | {'Edge SLOW_BEAR':<14} | {'Edge FAST_CRASH':<15} | {'Edge ALL':<8}")
print("-" * 105)
top_combos = df_combo[df_combo.index.str.contains(r'\+')].sort_values(by='edge_all', ascending=False).head(5)
for name, row in top_combos.iterrows():
    calm = row['edge_calm'] * 100
    slow = row['edge_slow_bear'] * 100
    fast = row['edge_fast_crash'] * 100
    all_e = row['edge_all'] * 100
    print(f"{name:<45} | {calm:>7.2f} pp | {slow:>11.2f} pp | {fast:>12.2f} pp | {all_e:>6.2f} pp")

print("\n\n2. TABLA DE RÉGIMEN PARA INDIVIDUALES CLAVE\n")
print(f"{'Indicador':<20} | {'Edge CALM':<10} | {'Edge SLOW_BEAR':<14} | {'Edge FAST_CRASH':<15}")
print("-" * 68)

key_inds = ["RSI_14 <=20", "EMA200_disc >=15", "BB_pctB <=0.2", "Stoch_K <=20"]
for name in key_inds:
    if name in df_single.index:
        row = df_single.loc[name]
        calm = row['edge_calm'] * 100
        slow = row['edge_slow_bear'] * 100
        fast = row['edge_fast_crash'] * 100
        print(f"{name:<20} | {calm:>7.2f} pp | {slow:>11.2f} pp | {fast:>12.2f} pp")
        
print("\n\n3. ¿Algún combo supera a EMA200_disc >= 15 en TODO?\n")
ema15 = df_single.loc["EMA200_disc >=15"]
ema_calm = ema15['edge_calm']
ema_slow = ema15['edge_slow_bear']
ema_fast = ema15['edge_fast_crash']
ema_all = ema15['edge_all']

super_combos = df_combo[df_combo.index.str.contains(r'\+')]
super_combos = super_combos[
    (super_combos['edge_calm'] > ema_calm) & 
    (super_combos['edge_slow_bear'] > ema_slow) & 
    (super_combos['edge_fast_crash'] > ema_fast)
]

if super_combos.empty:
    print("Ningún combo supera a EMA200_disc >= 15.0 simultáneamente en los 3 regímenes.")
else:
    print("¡SÍ! Los siguientes combos lo superan en TODOS los regímenes:")
    for name, row in super_combos.iterrows():
        print(f"- {name}")
