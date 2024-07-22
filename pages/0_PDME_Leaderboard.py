import streamlit as st
import openai
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
import pandas as pd
from pdme.generate_bootstrap_prompts import create_bootstrap_prompts
from pdme.evaluate import pdme_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

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
            response = model.generate_content(item)
            responses.append(response.text)

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

        avg_score = [(score_1[j] + score_2[j]) / 2 for j in range(len(score_1))]

        model_1_scores.append(avg_score[1])
        model_2_scores.append(avg_score[0])
        sum_model_1_scores += avg_score[1]
        sum_model_2_scores += avg_score[0]

    winner = "Model 1" if sum_model_1_scores > sum_model_2_scores else "Model 2"

    scores_dict = {
        "Model 1 Scores": model_1_scores,
        "Model 2 Scores": model_2_scores,
        "Model 1 Total Scores": sum_model_1_scores,
        "Model 2 Total Scores": sum_model_2_scores,
        "Winner": winner
    }

    return scores_dict

def rank_models(results, models):
    logging.info('Ranking models...')
    scores = {model: 0 for model in models}

    for model1, model2, winner in results:
        if winner == "Model 1":
            scores[model1] += 1
        elif winner == "Model 2":
            scores[model2] += 1

    leaderboard = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    leaderboard_df = pd.DataFrame(leaderboard, columns=["Model Name", "Wins"])
    leaderboard_df["Rank"] = leaderboard_df["Wins"].rank(ascending=False, method='dense').astype(int)

    return leaderboard_df

# Initialize Streamlit app
st.title('Model Evaluation App')

# Multiselect for models
model_list = ['claude-3-opus-20240229', 'claude-3-5-sonnet-20240620', 'gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gemini-1.0-pro']
selected_models = st.multiselect('Select models to evaluate:', model_list)

# Button to generate bootstrap prompts
if 'bootstrap_prompts' not in st.session_state:
    st.session_state.bootstrap_prompts = []
if st.button('Generate Bootstrap Prompts'):
    seeds = { 
        "<language>": ["python", "c++"],
        "<seed>": ["tic-tac-toe", "array", "sorting", "dictionary"],
    }
    bootstrap_prompt_template = """Write a question asking to make a programming challenge meant to evaluate programming abilities.
    The problem should be possible to solve in less than 100 lines of code for a very skilled programmer.
    The problem should use the <language> language, and be related to these seeds: <seed>, <seed>."""
    st.session_state.bootstrap_prompts = generate_bootstrap_prompts(seeds, bootstrap_prompt_template, num=3)
    st.write(st.session_state.bootstrap_prompts)

# Generate question prompts
if 'question_prompts' not in st.session_state:
    st.session_state.question_prompts = []
if st.button('Generate Question Prompts'):
    st.session_state.question_prompts = generate_question_prompts(st.session_state.bootstrap_prompts, model_name="gpt-3.5-turbo", api_key=openai_api_key)
    st.write(st.session_state.question_prompts)

# Logging area
if 'log' not in st.session_state:
    st.session_state.log = []
def clear_log():
    st.session_state.log = []
log_area = st.empty()

# Results DataFrame
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["Model 1", "Model 2", "Winner"])

# Run Next Pair button
if 'model_pair_index' not in st.session_state:
    st.session_state.model_pair_index = 0
if st.button('Run Next Pair'):
    clear_log()
    if st.session_state.model_pair_index < len(selected_models) - 1:
        i = st.session_state.model_pair_index
        model_1 = selected_models[i]
        model_2 = selected_models[i + 1]

        log_area.info(f'Running competition between {model_1} and {model_2}')

        responses_model_1 = generate_responses(model_1, st.session_state.question_prompts)
        responses_model_2 = generate_responses(model_2, st.session_state.question_prompts)

        eval_model = "gpt-3.5-turbo-instruct"
        client = openai.OpenAI(api_key=openai_api_key)

        evaluation_prompt_template = """<prefix><user_start>Here is a prompt:
        {
            "instruction": \"""<question_full>\""",
        }

        Here are the outputs of the models:
        [
            {
                "model": 1,
                "answer": \"""<response1>\"""
            },
            {
                "model": 2,
                "answer": \"""<response2>\"""
            }
        ]

        Please create a dictLet's continue with the modifications to complete the Streamlit application script.

### Streamlit Script (continued)

```python
        Please create a dict containing the highest quality answer, i.e., produce the following output:

        {
            'best_model': <model-name>
        }

        Please provide the response that the majority of humans would consider better.

        <assistant_start>{
        'best_model': """

        scores = score_responses(evaluation_prompt_template, st.session_state.question_prompts, responses_model_1, responses_model_2, client, eval_model)

        log_area.info(f'Scores: {scores}')
        winner = scores["Winner"]
        
        new_row = {"Model 1": model_1, "Model 2": model_2, "Winner": winner}
        st.session_state.results_df = st.session_state.results_df.append(new_row, ignore_index=True)
        
        st.session_state.model_pair_index += 1
    else:
        log_area.info('All pairs have been evaluated.')

# Display the results DataFrame
st.write(st.session_state.results_df)

# Rank Models button
if st.button('Rank Models'):
    leaderboard_df = rank_models(st.session_state.results_df.values, selected_models)
    st.write(leaderboard_df)

# Add a log output area
st.text_area("Log Output", value="\n".join(st.session_state.log), height=200)
