import pandas as pd
from collections import defaultdict

def compute_online_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)

    for rd, model_a, model_b, winner in battles[['Model 1', 'Model 2', 'Winner']].itertuples():
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == "Model 1":
            sa = 1
        elif winner == "Model 2":
            sa = 0
        elif winner == "tie" or winner == "tie (bothbad)":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    # calibrate llama-13b to 800
    delta = (800-rating["claude-3-opus-20240229"])
    for model in battles["Model 1"].unique():
        rating[model] += delta

    return rating

battles = pd.read_csv('data/generic-competitions.csv')
ratings = compute_online_elo(battles)
# Convert to DataFrame
elo_df = pd.DataFrame(list(ratings.items()), columns=['model_name', 'elo_ranking'])

# Sort by elo_ranking in descending order
elo_df = elo_df.sort_values(by='elo_ranking', ascending=False).reset_index(drop=True)
print(elo_df.head(10))
