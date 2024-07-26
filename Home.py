import streamlit as st
import openai
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
import pandas as pd
from itertools import combinations
from pdme.generate_bootstrap_prompts import create_bootstrap_prompts
from pdme.evaluate import pdme_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

template_file_path = "templates/evaluation_template.md"

# Function to load the markdown template
def load_template(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def load_questions(file_path):
    content = load_template(file_path)
    questions = content.split('\n')
    return [q.strip() for q in questions if q.strip()]

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

                # Average the scores
        # Now we know exactly which score corresponds to which model
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


# Initialize Streamlit app
st.title('PDME Arena')

st.markdown("""
## Overview

The method uses a single text generation AI, referred to as eval model, to evaluate any other text generation AI on any topic, and the evaluation works like this:

1. **Bootstrap prompt generation** -wWe write a text prompt for what questions the eval model should generate, 
            and provide seeds that are randomly picked to generate a question in various categories such as general questions, coding, maths and story telling.
2. **Questio generation** - we then generate a full-fledged question to be sent a pair of models to which each generate their responses.
3. **Evaluation and scoring** - we then evaluate the responses of the two models using the eval model to determine the winner.
5. For *n* models, *n(n-1)/2* comparisons are made to generate competition results.
6. Finally, we calculate ELO ranking to rank the models.

This method allows us to evaluate models for any topic, such as generic question,  storytelling, programming, and finance.
The Evaluator Model is currently always assumed to be OpenAI's GPT-3.5 Turbo Instruct.

## References - Available Models

* [OpenAI GPT Models](https://platform.openai.com/docs/models)
* [Anthropic Claude Models](https://docs.anthropic.com/en/docs/about-claude/models)
* [Google Gemini Models](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions)
* [HuggingFace Text Generation Models](https://huggingface.co/models?pipeline_tag=text-generation)
* [VLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html#supported-models)
* [LLM Arena Leaderboard](https://chat.lmsys.org/?leaderboard)

## Notes

* The Evaluator Model is currently always assumed to be OpenAI's GPT-3.5 Turbo Instruct.
* Some providers, such as OpenAI and Google point the latest model to a particular release (eg gpt-4o -> gpt-4o-2024-05-13) while for others such as Anthropic you have to hard code the release data / number into the model name (claude-3-5-sonnet-20240620).            

""")

# Multiselect for models
model_list = ['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229', 'gpt-4o-2024-05-13', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4-1106-preview', 'gemini-1.5-pro-api-0409-preview']
selected_models = st.multiselect('Select models to evaluate:', model_list, default=model_list)

# Select box for evaluation type
eval_type = st.selectbox('Select Evaluation Type', ['Generic', 'Coding', 'Story Telling'])
eval_type_var = eval_type.lower().replace(' ', '_')

# Slider to select the number of prompts to generate
num_prompts = st.selectbox('Select number of prompts to generate:', options=list(range(1, 11)), index=4)


# Generate all unique pairs of models
if 'model_pairs' not in st.session_state:
    st.session_state.model_pairs = list(combinations(selected_models, 2))

# Conditional generation of bootstrap prompts
if eval_type_var in ['coding', 'story_telling']:
    if 'bootstrap_prompts' not in st.session_state:
        st.session_state.bootstrap_prompts = []

    if st.button('Generate Bootstrap Prompts'):
        if eval_type_var == 'coding':
            seeds = { 
                "<language>": ["python", "c++"],
                "<seed>": ["tic-tac-toe", "array", "sorting", "dictionary"],
            }
            bootstrap_prompt_template = load_template('templates/coding_template.md')
        elif eval_type_var == 'story_telling':
            seeds = {
                "seed_1": ["a haunted house", "a time traveler", "a magical forest"],
                "seed_2": ["redemption", "discovery", "loss"],
                "seed_3": ["a talking animal", "an ancient artifact", "a secret society"],
                "seed_4": ["a plot twist", "a moral dilemma", "an unexpected friendship"]
            }
            bootstrap_prompt_template = load_template('templates/story_telling_template.md')

        st.session_state.bootstrap_prompts = generate_bootstrap_prompts(seeds, bootstrap_prompt_template, num=num_prompts)
        st.write(st.session_state.bootstrap_prompts)

# Generate question prompts
if 'question_prompts' not in st.session_state:
    st.session_state.question_prompts = []

if st.button('Generate Question Prompts'):
    if eval_type_var == 'generic':
        general_questions_file_path = 'templates/general_question_template.md'
        all_questions = load_questions(general_questions_file_path)
        st.session_state.question_prompts = all_questions[:num_prompts]
    else:
        st.session_state.question_prompts = generate_question_prompts(st.session_state.bootstrap_prompts, model_name="gpt-3.5-turbo", api_key=openai_api_key)
    
    st.write(st.session_state.question_prompts)

    if st.session_state.model_pair_index < len(st.session_state.model_pairs):
        model_1, model_2 = st.session_state.model_pairs[st.session_state.model_pair_index]
        st.success(f'Model 1: {model_1}, Model 2: {model_2}')

# Logging area
if 'log' not in st.session_state:
    st.session_state.log = []
def clear_log():
    st.session_state.log = []
log_area = st.empty()

# Results DataFrame
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["Model 1", "Model 2", "Winner"])

# Run Get Responses button
if 'model_pair_index' not in st.session_state:
    st.session_state.model_pair_index = 0
if st.button('Get Responses'):
    clear_log()
    if st.session_state.model_pair_index < len(st.session_state.model_pairs):
        i = st.session_state.model_pair_index
        model_1, model_2 = st.session_state.model_pairs[i]

        log_area.info(f'Generating responses for Model 1: {model_1} and Model 2: {model_2}')

        with st.spinner('Generating responses...'):
            st.session_state.responses_model_1 = generate_responses(model_1, st.session_state.question_prompts)
            st.session_state.responses_model_2 = generate_responses(model_2, st.session_state.question_prompts)

        st.write(f"Responses from Model 1 ('{model_1}'):")
        st.write(st.session_state.responses_model_1)
        st.write(f"Responses from Model 2 ('{model_2}'):")
        st.write(st.session_state.responses_model_2)

# Run Evaluate Next Pair button
if st.button('Score Responses'):
    clear_log()
    if st.session_state.model_pair_index < len(st.session_state.model_pairs):
        model_1, model_2 = st.session_state.model_pairs[st.session_state.model_pair_index]

        log_area.info(f'Evaluating competition between Model 1: {model_1} and Model 2: {model_2}')

        eval_model = "gpt-3.5-turbo-instruct"
        client = openai.OpenAI(api_key=openai_api_key)

        evaluation_prompt_template = load_template('templates/evaluation_template.md')

        scores = score_responses(evaluation_prompt_template, st.session_state.question_prompts, st.session_state.responses_model_1, st.session_state.responses_model_2, client, eval_model)

        log_area.info(f'Scores: {scores}')
        winner = scores["Winner"]
        
        new_row = pd.DataFrame({
            "Model 1": [model_1],
            "Model 2": [model_2],
            "Model 1 Total Score": [scores["Model 1 Total Score"]],
            "Model 2 Total Score": [scores["Model 2 Total Score"]],
            "Winner": [winner]
        }, index=[0])
        st.session_state.results_df = pd.concat([st.session_state.results_df.reset_index(drop=True), new_row.reset_index(drop=True)], ignore_index=True)

        st.session_state.model_pair_index += 1
    else:
        log_area.info('All pairs have been evaluated.')

# Display the results DataFrame
st.write(st.session_state.results_df)

