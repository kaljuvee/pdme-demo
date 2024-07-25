import argparse
import logging
import os
import pandas as pd
from itertools import combinations
from dotenv import load_dotenv

import openai
import anthropic
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError, RetryError

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
    content = load_template(file_path)
    if content:
        questions = content.split('\n')
        return [q.strip() for q in questions if q.strip()]
    return []

def generate_bootstrap_prompts(seeds, template, num):
    logging.info('Generating bootstrap prompts...')
    return create_bootstrap_prompts(template=template, seeds=seeds, num=num)

def generate_question_prompts(bootstrap_prompts, model_name, api_key):
    logging.info('Generating question prompts using model %s...', model_name)
    client = openai.OpenAI(api_key=api_key)
    question_prompts = []

    for item in bootstrap_prompts:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": item},
            ]
        )
        question_prompts.append(response.choices[0].message.content)
    
    return question_prompts

def generate_responses(model_name, question_prompts):
    logging.info('Generating responses using model %s...', model_name)
    responses = []

    if model_name.startswith("gpt"):
        client = openai.OpenAI(api_key=openai_api_key)
        for item in question_prompts:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": item},
                ]
            )
            responses.append(response.choices[0].message.content)

    elif model_name.startswith("claude"):
        anthropic_client = anthropic.Client(api_key=anthropic_api_key)
        for item in question_prompts:
            response = anthropic_client.messages.create(
                model=model_name,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": item}
                ]
            )
            text_response = response.content[0].text
            responses.append(text_response)

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

def score_responses(evaluation_prompt_template, question_prompts, responses_model_1, responses_model_2, client, eval_model):
    logging.info('Scoring responses...')
    llm = pdme_llm(client, eval_model)
    model_1_scores = []
    model_2_scores = []
    sum_model_1_scores = 0
    sum_model_2_scores = 0

    for i, question in enumerate(question_prompts):
        prompt_1 = evaluation_prompt_template.replace("<question_full>", question).replace("<response1>", responses_model_2[i]).replace("<response2>", responses_model_1[i])
        score_1 = llm.evaluate(prompt_1, ["1", "2"])

        prompt_2 = evaluation_prompt_template.replace("<question_full>", question).replace("<response1>", responses_model_1[i]).replace("<response2>", responses_model_2[i])
        score_2 = llm.evaluate(prompt_2, ["1", "2"])

        # Average the scores
        model_1_score = (score_1[1] + score_2[0]) / 2  # Average of Model 1's scores
        model_2_score = (score_1[0] + score_2[1]) / 2  # Average of Model 2's scores

        model_1_scores.append(model_1_score)
        model_2_scores.append(model_2_score)
        sum_model_1_scores += model_1_score
        sum_model_2_scores += model_2_score

    winner = "Model 1" if sum_model_1_scores > sum_model_2_scores else "Model 2"

    scores_dict = {
        "Model 1 Scores": model_1_scores,
        "Model 2 Scores": model_2_scores,
        "Model 1 Total Score": sum_model_1_scores,
        "Model 2 Total Score": sum_model_2_scores,
        "Winner": winner
    }

    return scores_dict

def main(models_file, eval_type, num_prompts, output_file):
    # Load models from CSV file
    models_df = pd.read_csv(models_file)
    model_list = models_df['model_name'].tolist()

    # Generate all unique pairs of models
    model_pairs = list(combinations(model_list, 2))

    # Pre-load questions based on evaluation type
    bootstrap_prompts = []
    question_prompts = []

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

    results_df = pd.DataFrame(columns=["Model 1", "Model 2", "Model 1 Total Score", "Model 2 Total Score", "Winner"])

    eval_model = "gpt-3.5-turbo-instruct"
    client = openai.OpenAI(api_key=openai_api_key)
    evaluation_prompt_template = load_template('templates/evaluation_template.md')

    for model_1, model_2 in model_pairs:
        logging.info(f'Generating responses for Model 1: {model_1} and Model 2: {model_2}')
        
        responses_model_1 = generate_responses(model_1, question_prompts)
        if responses_model_1 is None:
            logging.warning(f'Skipping evaluation for Model 1: {model_1} due to generation error.')
            continue
        responses_model_2 = generate_responses(model_2, question_prompts)
        if responses_model_2 is None:
            logging.warning(f'Skipping evaluation for Model 2: {model_2} due to generation error.')
            continue

        scores = score_responses(evaluation_prompt_template, question_prompts, responses_model_1, responses_model_2, client, eval_model)

        logging.info(f'Scores: {scores}')
        winner = scores["Winner"]
        
        new_row = pd.DataFrame({
            "Model 1": [model_1],
            "Model 2": [model_2],
            "Model 1 Total Score": [scores["Model 1 Total Score"]],
            "Model 2 Total Score": [scores["Model 2 Total Score"]],
            "Winner": [winner]
        }, index=[0])
        results_df = pd.concat([results_df.reset_index(drop=True), new_row.reset_index(drop=True)], ignore_index=True)
        logging.info(results_df)

    # Save the results to a CSV file
    results_df.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDME Arena evaluation.")
    parser.add_argument("--models_file", type=str, required=True, help="Path to the CSV file containing model names.")
    parser.add_argument("--eval_type", type=str, choices=["generic", "coding", "story_telling"], required=True, help="Type of evaluation.")
    parser.add_argument("--num_prompts", type=int, default=5, help="Number of prompts to generate.")
    parser.add_argument("--output_file", type=str, default="results.csv", help="Path to the output CSV file for results.")

    args = parser.parse_args()
    main(args.models_file, args.eval_type, args.num_prompts, args.output_file)
