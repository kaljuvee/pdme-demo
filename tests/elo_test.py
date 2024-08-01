import pandas as pd
from pdme.evaluate import pdme_llm
from utils import calculate_elo_iterative

battles = pd.read_csv('data/generic_battle-20240726-0900.csv')
print(battles.head())
#alibration_model = 'claude-3-opus-20240229'

#elo_df = pdme_llm.compute_online_elo(battles, calibration_model)
# Convert to DataFrame

#elo_df.to_csv('data/pdme_elo_generic.csv', index=False)

elo_iter_df = calculate_elo_iterative(battles)
elo_iter_df.to_csv('data/pdme_elo_generic_iterative.csv', index=False)

print(elo_iter_df.head())
