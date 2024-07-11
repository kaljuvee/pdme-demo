import streamlit as st
import logging
from dotenv import load_dotenv
from opti_pdme.opticonomy_pdme import PDME
import opti_pdme.model_utils as model_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load the markdown template
def load_template(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to generate question prompts using the evaluation model
def generate_question_prompts(eval_model, prompts):
    try:
        bound_llm = eval_model.bind(logprobs=True)
    except Exception as e:
        st.error(f"Error binding eval_model: {e}")
        return []
    
    question_prompts = []
    for item in prompts:
        try:
            st.info(f'Generating question prompt for bootstrap prompt: {item}')
            response = bound_llm.invoke(item)
            generated_text = response.content
            st.success(f"Generated question prompt: {generated_text.strip()}")
            question_prompts.append(generated_text.strip())
        except Exception as e:
            st.error(f"Error generating question prompt: {e}")
            question_prompts.append("")
    
    return question_prompts

# Main Streamlit app
def main():
    st.title("Prompt Generation and Evaluation App")

    # Template selection
    template_type = st.selectbox("Choose template type:", ["storytelling", "coding"])

    # Seed input
    st.subheader("Customize Seeds")
    if template_type == "storytelling":
        seeds = {
            "seed_1": st.text_input("seed_1", value="a haunted house, a time traveler, a magical forest"),
            "seed_2": st.text_input("seed_2", value="redemption, discovery, loss"),
            "seed_3": st.text_input("seed_3", value="a talking animal, an ancient artifact, a secret society"),
            "seed_4": st.text_input("seed_4", value="a plot twist, a moral dilemma, an unexpected friendship")
        }
        template_file_path = "templates/storytelling_template.md"
    else:
        seeds = {
            "<language>": st.text_input("<language>", value="python, c++"),
            "<seed>": st.text_input("<seed>", value="tic-tac-toe, array, sorting, dictionary")
        }
        template_file_path = "templates/coding_template.md"

    # Convert input strings to lists
    for key in seeds:
        seeds[key] = [item.strip() for item in seeds[key].split(",")]

    # Model selection
    st.subheader("Model Selection")
    eval_model_options = ["openai/gpt-4o", "openai/gpt-3.5-turbo", "anthropic/claude-v1"]  # Add more options as needed
    test_model_options = ["openai-community/gpt2", "facebook/opt-350m", "EleutherAI/gpt-neo-125M"]  # Add more options as needed
    
    eval_model_name = st.selectbox("Choose evaluation model:", eval_model_options)
    test_model_name = st.selectbox("Choose test model:", test_model_options)

    # Number of questions to generate
    num_questions = st.select_slider("Number of questions to generate:", options=range(1, 11), value=2)

    # Load template
    bootstrap_prompt_template = load_template(template_file_path)

    # Generate button
    if st.button("Generate"):
        # Load models
        eval_model = model_utils.load_model(eval_model_name)
        if eval_model is None:
            st.error("Failed to load the evaluation model.")
            return

        test_model, test_tokenizer = model_utils.load_model(test_model_name)
        if test_model is None:
            st.error(f"Failed to load the test model '{test_model_name}'.")
            return

        # Initialize the PDME evaluator
        pdme = PDME(eval_model, (test_model, test_tokenizer))

        # Generate the bootstrap prompts
        bootstrap_prompts = pdme.generate_bootstrap_prompt(bootstrap_prompt_template, seeds, num=num_questions, prompt_type=template_type)

        st.subheader("Generated Bootstrap Prompts")
        for i, prompt in enumerate(bootstrap_prompts, 1):
            st.text(f"Bootstrap Prompt {i}:\n{prompt}")

        # Generate the question prompts
        question_prompts = generate_question_prompts(eval_model, bootstrap_prompts)

        st.subheader("Evaluation Results")
        for i, item in enumerate(question_prompts, 1):
            st.write(f"Evaluating Prompt {i}")
            eval_response = pdme.generate_text(eval_model, item, max_new_tokens=2000)
            st.info(f'Eval response: {eval_response}')
            test_response = pdme.generate_text(test_model, item, max_new_tokens=2000)
            st.info(f'Test response: {test_response}')

            result, probabilities = pdme.evaluate(item, eval_response, test_response, eval_model_name, test_model_name)

            if result is not None:
                st.success(f"Evaluation result: {result}")
                st.json(probabilities)
            else:
                st.error("Evaluation failed.")

if __name__ == "__main__":
    main()