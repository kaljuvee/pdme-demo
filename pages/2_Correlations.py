import streamlit as st
import pandas as pd
from scipy.stats import pearsonr

def match_keys(df, key_column, match_dict):
    df[key_column] = df[key_column].replace(match_dict)
    return df

def compute_correlations(df1, df2):
    merged_df = pd.merge(df1, df2, on='model_name')
    merged_df = merged_df.rename(columns={'elo_ranking_x': 'elo_ranking_pdme', 'elo_ranking_y': 'elo_ranking_llmarena'})
    pearson_corr, pearson_p = pearsonr(merged_df['elo_ranking_pdme'], merged_df['elo_ranking_llmarena'])
    
    return {
        "pearson_corr": pearson_corr,
        "pearson_p": pearson_p,
        "merged_df": merged_df
    }

# Streamlit app
st.title('PDME vs LLM Arena Correlations')

st.markdown("""
## Upload ELO Results File
Upload the CSV file containing the ELO results.
The CSV file should have columns: model_name, elo_ranking.
""")

uploaded_elo_file = st.file_uploader("Choose a CSV file for ELO results", type="csv")

if uploaded_elo_file is not None:
    elo_results = pd.read_csv(uploaded_elo_file)
    st.write("### Uploaded ELO Results file:")
    st.write(elo_results.head())

st.markdown("""
## LLM Arena Data File
You can either upload a new CSV file or use the existing `llmarena_elo.csv` file.
""")

llmarena_default_path = 'data/llmarena_elo.csv'

use_default_llmarena = st.checkbox('Use existing LLM Arena data file (llmarena_elo.csv)')

if use_default_llmarena:
    llm_arena_data = pd.read_csv(llmarena_default_path)
    st.write("### Existing LLM Arena Data (First 10 rows):")
    st.write(llm_arena_data.head(10))
else:
    uploaded_llm_file = st.file_uploader("Choose a CSV file for LLM Arena data", type="csv")

    if uploaded_llm_file is not None:
        llm_arena_data = pd.read_csv(uploaded_llm_file)
        st.write("### Uploaded LLM Arena Data (First 10 rows):")
        st.write(llm_arena_data.head(10))

if uploaded_elo_file is not None and (use_default_llmarena or uploaded_llm_file is not None):
    if st.button("Calculate Correlation"):
        # Define match dictionary
        match_dict = {
            'gemini-1.5-pro-api-0409-preview': 'gemini-1.5-pro',
            'gpt-4-1106-preview': 'gpt-4'
        }
        
        # Match keys in both dataframes
        elo_results = match_keys(elo_results, 'model_name', match_dict)
        llm_arena_data = match_keys(llm_arena_data, 'model_name', match_dict)
        
        correlations = compute_correlations(elo_results, llm_arena_data)
        merged_df = correlations['merged_df']
        
        st.write("### Merged Data:")
        st.write(merged_df)

        st.write(f"Pearson correlation coefficient: {correlations['pearson_corr']} (p-value: {correlations['pearson_p']})")
   
        # Interpret results
        if correlations['pearson_corr'] > 0.7:
            pearson_interpretation = "high"
        elif correlations['pearson_corr'] > 0.5:
            pearson_interpretation = "moderate"
        else:
            pearson_interpretation = "low"

        st.write(f"The Pearson correlation is {pearson_interpretation} with a coefficient of {correlations['pearson_corr']}.")
