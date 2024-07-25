import streamlit as st
import pandas as pd

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
