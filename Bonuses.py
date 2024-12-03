import pandas as pd
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(layout='wide')

# Load data from URLs
matchstats_url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/Bonus.csv'
possession_data_url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/Horsens/Horsens_possession_data.csv'

matchstats_df = pd.read_csv(matchstats_url)
possession_data_df = pd.read_csv(possession_data_url)

# Filter for goals (typeId == 16) and select relevant columns
goals = possession_data_df[possession_data_df['typeId'] == 16]
goals = goals[['team_name', 'playerName', 'label', 'date']]

# List of players' goals
player_goal_counts = (
    goals.groupby('playerName')
    .size()
    .reset_index(name='goal_count')  # Rename the resulting column
)

# Group by match (label + date) and team to count goals
goal_counts = (
    goals.groupby(["label", "date", "team_name"])
    .size()
    .reset_index(name="team_goals")
)

# Add total goals in the match for each label and date
goal_counts['goals_in_match'] = goal_counts.groupby(['label', 'date'])['team_goals'].transform('sum')

# Calculate the result for each team
goal_counts['result'] = goal_counts['team_goals'] - (goal_counts['goals_in_match'] - goal_counts['team_goals'])

# Determine if the team won the match
goal_counts['win'] = goal_counts['result'] > 0

# Filter for Horsens matches and relevant columns
horsens_results = goal_counts[goal_counts['team_name'] == 'Horsens']
horsens_results = horsens_results[['label', 'date', 'win']]

# Display the dataframes in Streamlit
print(player_goal_counts)  # Players' goals
print(horsens_results)  # Match results for Horsens
matchstats_df = matchstats_df[['player_matchName','label','date','player_position','minsPlayed']]
# Assuming matchstats_df is your dataframe
aggregated_df = (
    matchstats_df.groupby(['player_matchName','label'])
    .agg(
        In_squad=('player_matchName', 'count'),  # Count appearances of each player
        Starting_11=('player_position', lambda x: (x != 'Substitute').sum()),  # Count non-substitute entries
        total_minutes_played=('minsPlayed', 'sum')
    )
)

# Set the index as player_matchName (this is already the case after groupby)
aggregated_df.index.name = 'player_matchName'
print(aggregated_df)
st.dataframe(aggregated_df)