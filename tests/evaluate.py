import math
import json
from collections import defaultdict
import pandas as pd
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class pdme_llm:
    """
    A class to interact with language models for evaluation and ELO ranking computation.

    Attributes:
        client (object): The client object for the language model API.
        model_name (str): The name of the model to use.
        is_chat (bool): Indicates if the model is a chat-based model.

    Methods:
        evaluate(prompt, labels): Evaluates the given prompt and returns label probabilities.
        generate(input, max_tokens, logprobs=5): Generates a response from the model.
        probability_of_labels(response, labels): Computes the probability of given labels from the model response.
        compute_online_elo(battles, calibration_model, K=4, SCALE=400, BASE=10, INIT_RATING=1000): Computes ELO rankings based on model battles.
    """

    def __init__(self, client, model_name, is_chat=False):
        """
        Initializes the pdme_llm object.

        Args:
            client (object): The client object for the language model API.
            model_name (str): The name of the model to use.
            is_chat (bool, optional): Indicates if the model is a chat-based model. Defaults to False.
        """
        self.client = client
        self.model_name = model_name
        self.is_chat = is_chat

    def evaluate(self, prompt, labels):
        """
        Evaluates the given prompt and returns label probabilities.

        Args:
            prompt (str): The prompt to be evaluated.
            labels (list): The list of labels to evaluate.

        Returns:
            list: A list of probabilities corresponding to the labels.
        """
        response = self.generate(prompt, 1)
        if response is None:
            logging.error(f"No response generated for prompt: {prompt}")
            return [0] * len(labels)
        label_probabilities = self.probability_of_labels(response, labels)
        return label_probabilities

    def generate(self, input, max_tokens, logprobs=5):
        """
        Generates a response from the model.

        Args:
            input (str): The input prompt or messages for the model.
            max_tokens (int): The maximum number of tokens to generate.
            logprobs (int, optional): The number of log probabilities to return. Defaults to 5.

        Returns:
            object: The response from the model.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return None

    def probability_of_labels(self, response, labels):
        """
        Computes the probability of given labels from the model response.

        Args:
            response (object): The response from the model.
            labels (list): The list of labels to compute probabilities for.

        Returns:
            list: A list of probabilities corresponding to the labels.
        """
        response = json.loads(response.json())

        logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]

        # Convert log probabilities to logits (using some math tricks)
        def logprob_to_logit(lp):
            return lp - math.log(1.0 - math.exp(lp))

        # Convert logits back to probabilities
        def logit_to_prob(l):
            e_l = math.exp(l)
            return e_l / (1.0 + e_l)

        label_probabilities = []
        for label in labels:
            curr_logprob = logprobs.get(label, None)
            curr_logit = logprob_to_logit(curr_logprob) if curr_logprob is not None else 0

            # Apply temperature scaling
            temperature = 3.0  # Adjust as needed. Higher values make predictions less extreme.
            curr_logit /= temperature

            curr_prob = logit_to_prob(curr_logit)

            label_probabilities.append(curr_prob)

        # Normalize the probabilities to sum to 100%
        total_prob = sum(label_probabilities)

        label_probabilities = [prob / total_prob for prob in label_probabilities]

        return label_probabilities

    @staticmethod
    def compute_online_elo(battles, calibration_model, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
        """
        Computes ELO rankings based on model battles.

        Args:
            battles (DataFrame): A DataFrame containing model battles with columns ['model_a', 'model_b', 'winner'].
            calibration_model (str): The model to calibrate the ELO scores to.
            K (int, optional): The K-factor for ELO computation. Defaults to 4.
            SCALE (int, optional): The scale factor for ELO computation. Defaults to 400.
            BASE (int, optional): The base for ELO computation. Defaults to 10.
            INIT_RATING (int, optional): The initial rating for models. Defaults to 1000.

        Returns:
            DataFrame: A DataFrame containing models and their computed ELO rankings.
        """
        rating = defaultdict(lambda: INIT_RATING)

        for model_a, model_b, winner in battles[['model_a', 'model_b', 'winner']].itertuples(index=False):
            ra = rating[model_a]
            rb = rating[model_b]
            ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
            eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
            if winner == "model_a":
                sa = 1
            elif winner == "model_b":
                sa = 0
            elif winner == "tie" or winner == "tie (bothbad)":
                sa = 0.5
            else:
                raise Exception(f"unexpected vote {winner}")
            rating[model_a] += K * (sa - ea)
            rating[model_b] += K * (1 - sa - eb)

        # Calibrate the specified model to 800
        delta = (800 - rating[calibration_model])
        for model in battles["model_a"].unique():
            rating[model] += delta

        elo_df = pd.DataFrame(list(rating.items()), columns=['model_name', 'elo_ranking'])
        elo_df = elo_df.sort_values(by='elo_ranking', ascending=False).reset_index(drop=True)

        return elo_df

    @staticmethod
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
