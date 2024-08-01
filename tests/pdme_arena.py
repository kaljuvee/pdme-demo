import argparse
import logging
import os
import pandas as pd
from itertools import combinations
from dotenv import load_dotenv
from datetime import datetime
import openai
import anthropic
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError, RetryError
from scipy.stats import pearsonr
from utils import calculate_elo_iterative

from pdme.generate_bootstrap_prompts import create_bootstrap_prompts
from pdme.evaluate import pdme_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

template_file_path = "templates/evaluation_template.md"

# Function to load the markdown template
def load_template(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"Template file not found: {file_path}")
        return ""

def load_questions(file_path):
    try:
        # Try to load the DataFrame from the specified Parquet file
        df = pd.read_parquet("hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
        logging.info(f"Loaded DataFrame from Parquet file with shape: {df.shape}")

        # Log unique values in 'input' column to inspect
        unique_inputs = df['input'].unique()
        logging.info(f"Unique values in 'input' column: {unique_inputs}")

        # Only grab records where 'input' is empty or None
        df_filtered = df[df['input'].isnull()].head(100)
        logging.info(f"Filtered DataFrame with null 'input' and head 100: {df_filtered.head(5)}")

        questions = df_filtered['instruction'].tolist()
        if questions:
            logging.info(f"Loaded questions from Parquet file. Total questions loaded: {len(questions)}")
        else:
            logging.error("No questions loaded from Parquet file.")
            questions = []
    except Exception as e:
        logging.error(f"Error loading Parquet file: {e}")
        questions = []

    # Fallback to loading the template if no questions are loaded
    if not questions:
        content = load_template(file_path)
        if content:
            questions = content.split('\n')
            questions = [q.strip() for q in questions if q.strip()]
            logging.info(f"Loaded questions from template. Total questions loaded: {len(questions)}")
        else:
            questions = []

    return questions


def generate_bootstrap_prompts(seeds, template, num):
    logging.info('Generating bootstrap prompts...')
    return create_bootstrap_prompts(template=template, seeds=seeds, num=num)

def generate_question_prompts(bootstrap_prompts, model_name, api_key):
    logging.info('Generating question prompts using model %s...', model_name)
    client = openai.OpenAI(api_key=api_key)
    question_prompts = []

    for item in bootstrap_prompts:
        try:
            response = client.chat_completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": item},
                ]
            )
            question_prompts.append(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error generating question prompt for model {model_name}: {e}")
    
    return question_prompts

def generate_responses(model_name, question_prompts):
    logging.info('Generating responses using model %s...', model_name)
    responses = []

    if model_name.startswith("gpt"):
        client = openai.OpenAI(api_key=openai_api_key)
        for item in question_prompts:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": item},
                    ]
                )
                responses.append(response.choices[0].message.content)
            except Exception as e:
                logging.error(f"Error generating response for model {model_name}: {e}")

    elif model_name.startswith("claude"):
        anthropic_client = anthropic.Client(api_key=anthropic_api_key)
        for item in question_prompts:
            try:
                response = anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": item}
                    ]
                )
                text_response = response.content[0].text
                responses.append(text_response)
            except Exception as e:
                logging.error(f"Error generating response for model {model_name}: {e}")

    elif model_name.startswith("gemini"):
        genai.configure(api_key=google_api_key)
        model_last_part = model_name.split('/')[-1]
        logging.info('Generating for Gemini with: %s', model_last_part)
        model = genai.GenerativeModel(model_last_part)
        for item in question_prompts:
            try:
                response = model.generate_content(item)
                responses.append(response.text)
            except (GoogleAPICallError, RetryError, ValueError) as e:
                logging.error(f"Error generating response for {model_name}: {e}")
                return None

    else:
        raise ValueError(f"Unsupported model name '{model_name}'.")

    return responses

def score_responses(evaluation_prompt_template, question_prompts, responses_model_a, responses_model_b, client, eval_model):
    logging.info('Scoring responses...')
    llm = pdme_llm(client, eval_model)
    model_a_scores = []
    model_b_scores = []
    sum_model_a_scores = 0
    sum_model_b_scores = 0

    if not responses_model_a or not responses_model_b:
        logging.error("Empty responses detected. Skipping scoring.")
        return {
            "model_a_scores": model_a_scores,
            "model_b_scores": model_b_scores,
            "model_a_total_score": sum_model_a_scores,
            "model_b_total_score": sum_model_b_scores,
            "winner": "model_b" if sum_model_b_scores >= sum_model_a_scores else "model_a"
        }

    for i, question in enumerate(question_prompts):
        try:
            prompt_1 = evaluation_prompt_template.replace("<question_full>", question).replace("<response1>", responses_model_b[i]).replace("<response2>", responses_model_a[i])
            score_1 = llm.evaluate(prompt_1, ["1", "2"])

            prompt_2 = evaluation_prompt_template.replace("<question_full>", question).replace("<response1>", responses_model_a[i]).replace("<response2>", responses_model_b[i])
            score_2 = llm.evaluate(prompt_2, ["1", "2"])

            # Average the scores
            model_a_score = (score_1[1] + score_2[0]) / 2  # Average of Model A's scores
            model_b_score = (score_1[0] + score_2[1]) / 2  # Average of Model B's scores

            model_a_scores.append(model_a_score)
            model_b_scores.append(model_b_score)
            sum_model_a_scores += model_a_score
            sum_model_b_scores += model_b_score
        except Exception as e:
            logging.error(f"Error scoring responses for question {i}: {e}")

    winner = "model_a" if sum_model_a_scores > sum_model_b_scores else "model_b"

    scores_dict = {
        "model_a_scores": model_a_scores,
        "model_b_scores": model_b_scores,
        "model_a_total_score": sum_model_a_scores,
        "model_b_total_score": sum_model_b_scores,
        "winner": winner
    }

    return scores_dict

def compute_correlations(df1, df2):
    merged_df = pd.merge(df1, df2, on='model_name')
    merged_df = merged_df.rename(columns={'elo_ranking_x': 'elo_ranking_pdme', 'elo_ranking_y': 'elo_ranking_llmarena'})
    pearson_corr, pearson_p = pearsonr(merged_df['elo_ranking_pdme'], merged_df['elo_ranking_llmarena'])
    
    return {
        "pearson_corr": pearson_corr,
        "pearson_p": pearson_p,
        "merged_df": merged_df
    }

def save_results(results_df, battles_output_file):
    try:
        now = datetime.now()
        timestamp = now.strftime("-%Y%mdd-%H%M")
        battles_output_file_with_timestamp = f"{battles_output_file.rstrip('.csv')}{timestamp}.csv"
        results_df.to_csv(battles_output_file_with_timestamp, index=False)
        logging.info(f"Results saved to {battles_output_file_with_timestamp}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")

def save_elo_rankings(elo_df, iter_elo_df, elo_output_file):
    try:
        now = datetime.now()
        timestamp = now.strftime("-%Y%mdd-%H%M")
        elo_output_file_with_timestamp = f"{elo_output_file.rstrip('.csv')}{timestamp}.csv"
        iter_elo_output_file_with_timestamp = f"{elo_output_file.rstrip('.csv')}{timestamp}-iter.csv"

        elo_df.to_csv(elo_output_file_with_timestamp, index=False)
        logging.info(f"Calibrated ELO rankings saved to {elo_output_file_with_timestamp}")

        iter_elo_df.to_csv(iter_elo_output_file_with_timestamp, index=False)
        logging.info(f"Iterative ELO rankings saved to {iter_elo_output_file_with_timestamp}")
    except Exception as e:
        logging.error(f"Error saving ELO rankings: {e}")

def evaluate_models(model_pairs, question_prompts, evaluation_prompt_template, client, eval_model, results_df):
    for model_a, model_b in model_pairs:
        logging.info(f'Generating responses for model_a: {model_a} and model_b: {model_b}')
        
        try:
            responses_model_a = generate_responses(model_a, question_prompts)
            if responses_model_a is None:
                logging.warning(f'Skipping evaluation for model_a: {model_a} due to generation error.')
                continue
        except openai.error.InvalidRequestError as e:
            logging.error(f"Error generating responses for {model_a}: {e}")
            continue

        try:
            responses_model_b = generate_responses(model_b, question_prompts)
            if responses_model_b is None:
                logging.warning(f'Skipping evaluation for model_b: {model_b} due to generation error.')
                continue
        except openai.error.InvalidRequestError as e:
            logging.error(f"Error generating responses for {model_b}: {e}")
            continue
        
        logging.info(f"Responses for model_a {model_a}: {responses_model_a[:5]}")
        logging.info(f"Responses for model_b {model_b}: {responses_model_b[:5]}")

        scores = score_responses(evaluation_prompt_template, question_prompts, responses_model_a, responses_model_b, client, eval_model)

        logging.info(f'Scores: {scores}')
        winner = scores["winner"]
        
        new_row = pd.DataFrame({
            "model_a": [model_a],
            "model_b": [model_b],
            "model_a_total_score": [scores["model_a_total_score"]],
            "model_b_total_score": [scores["model_b_total_score"]],
            "winner": [winner]
        }, index=[0])
        results_df = pd.concat([results_df.reset_index(drop=True), new_row.reset_index(drop=True)], ignore_index=True)
        logging.info(results_df)

def setup_prompts(eval_type, num_prompts, openai_api_key):
    if eval_type in ['coding', 'story_telling']:
        if eval_type == 'coding':
            seeds = { 
                "<language>": ["python", "c++"],
                "<seed>": ["tic-tac-toe", "array", "sorting", "dictionary"],
            }
            bootstrap_prompt_template = load_template('templates/coding_template.md')
        elif eval_type == 'story_telling':
            seeds = {
                "seed_1": ["a haunted house", "a time traveler", "a magical forest"],
                "seed_2": ["redemption", "discovery", "loss"],
                "seed_3": ["a talking animal", "an ancient artifact", "a secret society"],
                "seed_4": ["a plot twist", "a moral dilemma", "an unexpected friendship"]
            }
            bootstrap_prompt_template = load_template('templates/story_telling_template.md')

        bootstrap_prompts = generate_bootstrap_prompts(seeds, bootstrap_prompt_template, num=num_prompts)
        question_prompts = generate_question_prompts(bootstrap_prompts, model_name="gpt-3.5-turbo", api_key=openai_api_key)

    elif eval_type == 'generic':
        general_questions_file_path = 'templates/general_question_template.md'
        all_questions = load_questions(general_questions_file_path)
        question_prompts = all_questions[:num_prompts]

    return question_prompts

def initialize_results_df():
    return pd.DataFrame(columns=["model_a", "model_b", "model_a_total_score", "model_b_total_score", "winner"])

def generate_model_pairs(model_list, eval_model, battle_type):
    if battle_type == "all_vs_all":
        return list(combinations(model_list, 2))
    elif battle_type == "eval_to_all":
        return [(eval_model, model) for model in model_list if model != eval_model]
    else:
        raise ValueError(f"Unsupported battle type '{battle_type}'.")

def main(models_file, eval_type, num_prompts, battles_output_file, elo_output_file, elo_calibration_model, elo_benchmark_file, eval_model, battle_type):
    # Load models from CSV file
    models_df = pd.read_csv(models_file)
    model_list = models_df['model_name'].tolist()

    # Generate model pairs based on the battle type
    model_pairs = generate_model_pairs(model_list, eval_model, battle_type)

    # Set up prompts
    question_prompts = setup_prompts(eval_type, num_prompts, openai_api_key)

    if not question_prompts:
        logging.error("No question prompts generated. Exiting.")
        return

    # Initialize results DataFrame
    results_df = initialize_results_df()

    # Load evaluation model and client
    client = openai.OpenAI(api_key=openai_api_key)
    evaluation_prompt_template = load_template('templates/evaluation_template.md')

    # Evaluate models
    evaluate_models(model_pairs, question_prompts, evaluation_prompt_template, client, eval_model, results_df)

    # Modify battles_output_file to include battle_type
    battles_output_file_with_type = battles_output_file.replace(".csv", f"_{battle_type}.csv")

    # Save results
    save_results(results_df, battles_output_file_with_type)

    # Compute benchmarked ELO rankings
    elo_df = pdme_llm.compute_online_elo(results_df, elo_calibration_model)
    iter_elo_df = calculate_elo_iterative(results_df)
    save_elo_rankings(elo_df, iter_elo_df, elo_output_file)

    # Calculate correlations for calibrated ELO
    llm_arena_data = pd.read_csv(elo_benchmark_file)
    correlations = compute_correlations(elo_df, llm_arena_data)
    logging.info(f"Calibrated ELO: Pearson correlation coefficient: {correlations['pearson_corr']} (p-value: {correlations['pearson_p']})")

    # Calculate correlations for iterative ELO
    iter_correlations = compute_correlations(iter_elo_df, llm_arena_data)
    logging.info(f"Iterative ELO: Pearson correlation coefficient: {iter_correlations['pearson_corr']} (p-value: {iter_correlations['pearson_p']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDME Arena evaluation.")
    parser.add_argument("--models_file", type=str, required=True, help="Path to the CSV file containing model names.")
    parser.add_argument("--eval_type", type=str, choices=["generic", "coding", "story_telling"], required=True, help="Type of evaluation.")
    parser.add_argument("--num_prompts", type=int, default=5, help="Number of prompts to generate.")
    parser.add_argument("--battles_output_file", type=str, default="data/generic_battles.csv", help="Path to the output CSV file for battle results.")
    parser.add_argument("--elo_output_file", type=str, default="data/generic_elo.csv", help="Path to the output CSV file for elo rankings.")
    parser.add_argument("--elo_calibration_model", type=str, default="claude-3-opus-20240229", help="ELO calibration model.")
    parser.add_argument("--elo_benchmark_file", type=str, default="llmarena_elo.csv", help="ELO benchmark file to correlate to.")
    parser.add_argument("--eval_model", type=str, default="gpt-3.5-turbo-instruct", help="Evaluation model.")
    parser.add_argument("--battle_type", type=str, choices=["all_vs_all", "eval_to_all"], required=True, help="Type of battle.")
    args = parser.parse_args()
    main(args.models_file, args.eval_type, 
         args.num_prompts, args.battles_output_file, 
         args.elo_output_file, 
         args.elo_calibration_model, args.elo_benchmark_file,
         args.eval_model, args.battle_type)
