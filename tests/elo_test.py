import pandas as pd
from pdme.evaluate import pdme_llm

battles = pd.read_csv('data/generic_battles.csv')

calibration_model = 'claude-3-opus-20240229'

elo_df = pdme_llm.compute_online_elo(battles, calibration_model)
# Convert to DataFrame

elo_df.to_csv('data/pdme_elo_generic.csv', index=False)
print(elo_df.head())
