import pandas as pd
import logging
from pdme_arena import load_questions, generate_responses, compute_correlations

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_load_questions():
    # Call the function
    general_questions_file_path = 'templates/general_question_template.md'
    questions = load_questions(general_questions_file_path)

    # Print the results
    print(f"Number of questions loaded: {len(questions)}")
    print("First 5 questions:")
    for i, question in enumerate(questions[:5], 1):
        print(f"{i}. {question}")

    # You can add more specific checks here if needed
    assert len(questions) > 0, "No questions were loaded"

# Test script
def test_generate_responses():
    # Define a set of question prompts
    question_prompts = [
        "What is the capital of France?",
        "Can you explain the theory of relativity?",
        "What are the benefits of using renewable energy sources?"
    ]

    # Test with gpt-3.5-turbo-instruct model
    model_name_gpt_instruct = "gpt-3.5-turbo-instruct"
    responses_gpt_instruct = generate_responses(model_name_gpt_instruct, question_prompts)
    logging.info('Responses from %s: %s', model_name_gpt_instruct, responses_gpt_instruct)

    # Test with gpt-4 (assuming gpt-4 is the chat model)
    model_name_gpt_chat = "gpt-4"
    responses_gpt_chat = generate_responses(model_name_gpt_chat, question_prompts)
    logging.info('Responses from %s: %s', model_name_gpt_chat, responses_gpt_chat)

    # Test with claude model (replace with actual model name if available)
    model_name_claude = "claude-3-opus-20240229"
    responses_claude = generate_responses(model_name_claude, question_prompts)
    logging.info('Responses from %s: %s', model_name_claude, responses_claude)

    # Test with gemini model (replace with actual model name if available)
    model_name_gemini = "gemini-1.5-pro"
    responses_gemini = generate_responses(model_name_gemini, question_prompts)
    logging.info('Responses from %s: %s', model_name_gemini, responses_gemini)

# Test function
def test_compute_correlations():
    # Create sample DataFrames
    df1  = pd.read_csv('data/llmarena_elo.csv')
    df2  = pd.read_csv('data/pdme_elo_generic.csv')
    
    # Compute correlations
    results = compute_correlations(df1, df2)
    
    # Print results for verification
    print("LLMArena vs ELO: Pearson correlation:", results["pearson_corr"])
    print("P-value:", results["pearson_p"])
    print("Merged DataFrame:\n", results["merged_df"])

    df2  = pd.read_csv('data/pdme_elo_generic_iterative.csv')
    results = compute_correlations(df1, df2)
    # Print results for verification
    print("LLMArena vs ELO Iterative: Pearson correlation:", results["pearson_corr"])
    print("P-value:", results["pearson_p"])
    print("Merged DataFrame:\n", results["merged_df"])
    
if __name__ == "__main__":
    # test_load_questions()
    #test_generate_responses()
    test_compute_correlations()
