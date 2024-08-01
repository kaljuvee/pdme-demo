import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_elo_iterative(df, initial_k=32, iterations=100, tolerance=0.1):
    try:
        matches = df.to_dict('records')
        ratings = {}

        def get_rating(model):
            if model not in ratings:
                ratings[model] = 1500  # Starting rating increased to 1500
            return ratings[model]

        def has_converged(old_ratings, new_ratings, tolerance):
            return all(abs(old_ratings[model] - new_ratings[model]) <= tolerance for model in old_ratings)

        def calculate_k(rating, games_played):
            # Dynamic K-factor that decreases as rating increases and games played increases
            base_k = initial_k
            rating_factor = max(1, (2000 - rating) / 200)
            games_factor = max(1, 30 / (games_played + 1))
            return base_k * rating_factor * games_factor

        games_played = {model: 0 for model in set(df['model_a']).union(set(df['model_b']))}

        for iteration in range(iterations):
            old_ratings = ratings.copy()
            new_ratings = old_ratings.copy()

            for match in matches:
                model_a, model_b = match['model_a'], match['model_b']
                winner = model_a if match['winner'] == 'model_a' else model_b
                loser = model_b if winner == model_a else model_a

                rating_a, rating_b = get_rating(model_a), get_rating(model_b)
                
                # Calculate score based on the total column
                score_a = match['model_a_total'] / (match['model_a_total'] + match['model_b_total'])
                score_b = 1 - score_a

                # Expected scores
                expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
                expected_b = 1 - expected_a

                # Dynamic K-factors
                k_a = calculate_k(rating_a, games_played[model_a])
                k_b = calculate_k(rating_b, games_played[model_b])

                # Update ratings
                new_ratings[model_a] = rating_a + k_a * (score_a - expected_a)
                new_ratings[model_b] = rating_b + k_b * (score_b - expected_b)

                # Increment games played
                games_played[model_a] += 1
                games_played[model_b] += 1

            logging.info(f"Iteration {iteration + 1}: {new_ratings}")

            if has_converged(old_ratings, new_ratings, tolerance):
                ratings = new_ratings
                logging.info(f"Converged after {iteration + 1} iterations.")
                break

            ratings = new_ratings
        
        # Normalize ratings to have a mean of 1500
        mean_rating = sum(ratings.values()) / len(ratings)
        normalized_ratings = {model: int(round(1500 + (rating - mean_rating))) for model, rating in ratings.items()}
        
        ratings_df = pd.DataFrame(list(normalized_ratings.items()), columns=['Model', 'Rating'])
        return ratings_df.sort_values('Rating', ascending=False).reset_index(drop=True)

    except KeyError as e:
        logging.error(f"Key error: {e}. Please ensure the DataFrame contains the correct columns.")
        return pd.DataFrame(columns=['Model', 'Rating'])
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return pd.DataFrame(columns=['Model', 'Rating'])