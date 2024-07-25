import streamlit as st
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def compute_correlations(df1, df2):
    merged_df = pd.merge(df1, df2, on='model_name')
    pearson_corr, pearson_p = pearsonr(merged_df['elo_ranking_x'], merged_df['elo_ranking_y'])
    spearman_corr, spearman_p = spearmanr(merged_df['elo_ranking_x'], merged_df['elo_ranking_y'])
    
    return {
        "pearson_corr": pearson_corr,
        "pearson_p": pearson_p,
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
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
## Upload LLM Arena Data File
Upload the CSV file containing the LLM Arena battles data.
""")

uploaded_llm_file = st.file_uploader("Choose a CSV file for LLM Arena data", type="csv")

if uploaded_llm_file is not None:
    llm_arena_data = pd.read_csv(uploaded_llm_file)
    st.write("### Uploaded LLM Arena Data (First 10 rows):")
    st.write(llm_arena_data.head(10))

if uploaded_elo_file is not None and uploaded_llm_file is not None:
    if st.button("Calculate Correlation"):
        correlations = compute_correlations(elo_results, llm_arena_data)
        merged_df = correlations['merged_df']
        
        st.write("### Merged Data:")
        st.write(merged_df)

        st.write(f"Pearson correlation coefficient: {correlations['pearson_corr']} (p-value: {correlations['pearson_p']})")
        st.write(f"Spearman correlation coefficient: {correlations['spearman_corr']} (p-value: {correlations['spearman_p']})")

        # Interpret results
        if correlations['pearson_corr'] > 0.7:
            pearson_interpretation = "high"
        elif correlations['pearson_corr'] > 0.5:
            pearson_interpretation = "moderate"
        else:
            pearson_interpretation = "low"

        if correlations['spearman_corr'] > 0.7:
            spearman_interpretation = "high"
        elif correlations['spearman_corr'] > 0.5:
            spearman_interpretation = "moderate"
        else:
            spearman_interpretation = "low"

        st.write(f"The Pearson correlation is {pearson_interpretation} with a coefficient of {correlations['pearson_corr']}.")
        st.write(f"The Spearman correlation is {spearman_interpretation} with a coefficient of {correlations['spearman_corr']}.")
