import streamlit as st
import pandas as pd
import plotly.express as px
from collections import defaultdict

# Define the ELO computation function
def compute_online_elo(battles, calibration_model, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)

    for model_a, model_b, winner in battles[['Model 1', 'Model 2', 'Winner']].itertuples(index=False):
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == "Model 1":
            sa = 1
        elif winner == "Model 2":
            sa = 0
        elif winner == "tie" or winner == "tie (bothbad)":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    # calibrate the specified model to 800
    delta = (800 - rating[calibration_model])
    for model in battles["Model 1"].unique():
        rating[model] += delta

    elo_df = pd.DataFrame(list(rating.items()), columns=['model_name', 'elo_ranking'])
    elo_df = elo_df.sort_values(by='elo_ranking', ascending=False).reset_index(drop=True)

    return elo_df

# Win Count Bar Chart
def plot_win_count(battles):
    # Count wins for each model
    model1_wins = battles[battles['Winner'] == 'Model 1']['Model 1'].value_counts()
    model2_wins = battles[battles['Winner'] == 'Model 2']['Model 2'].value_counts()
    
    # Combine wins
    total_wins = model1_wins.add(model2_wins, fill_value=0)
    
    # Handle ties
    ties = battles[battles['Winner'].isin(['tie', 'tie (bothbad)'])]
    for _, row in ties.iterrows():
        total_wins[row['Model 1']] += 0.5
        total_wins[row['Model 2']] += 0.5
    
    # Create DataFrame
    win_counts = pd.DataFrame({'Model': total_wins.index, 'Count': total_wins.values})
    
    # Sort by Count descending
    win_counts = win_counts.sort_values('Count', ascending=False).reset_index(drop=True)
    
    # Debugging: Display the win counts
    st.write("Win Counts:", win_counts)

    # Plot
    fig = px.bar(win_counts, x='Model', y='Count', title='Win Count for Each Model')
    fig.update_layout(xaxis_title="Model", yaxis_title="Count", height=400)

    return fig

# Scatter Plot for ELO Ratings
def plot_elo_ratings(elo_df):
    fig = px.scatter(elo_df, x='model_name', y='elo_ranking', title='ELO Ratings for Each Model', labels={'model_name': 'Model', 'elo_ranking': 'ELO Rating'})
    return fig

# Streamlit app
st.title('PDME Competition Data Visualizations')

st.markdown("""
## Upload CSV File
Upload the CSV file containing the battle results.
The CSV file should have columns: Model 1, Model 2, Winner.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    battles = pd.read_csv(uploaded_file)
    st.write("### Uploaded CSV file:")
    st.write(battles.head())

    # Get unique models for calibration dropdown
    unique_models = battles["Model 1"].unique()
    calibration_model = st.selectbox("Select Calibration Model", unique_models, index=0)

    if st.button('Rank'):
        elo_df = compute_online_elo(battles, calibration_model)
        st.session_state['elo_df'] = elo_df
        st.write("### ELO Rankings:")
        st.write(elo_df)
        st.markdown("**Observations**: The ELO rankings are displayed above. The x-axis represents the model names, and the y-axis shows the ELO ratings. Note that the calibration model was adjusted to an ELO rating of 800.")

    if st.button('Plot Win Count'):
        fig = plot_win_count(battles)
        st.plotly_chart(fig)
        st.markdown("**Observations**: The bar chart above shows the number of wins for each model. The x-axis represents the model names, and the y-axis shows the win count.")

    # Plot ELO Ratings if they exist in session state
    if 'elo_df' in st.session_state and st.button('Plot ELO Ratings'):
        fig5 = plot_elo_ratings(st.session_state['elo_df'])
        st.plotly_chart(fig5)
        st.markdown("**Observations**: This scatter plot shows the ELO ratings for each model. The x-axis represents the model names, and the y-axis shows the ELO ratings.")
