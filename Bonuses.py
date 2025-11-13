import pandas as pd
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(layout='wide')

# Load data from URLs
matchstats_url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/Bonus.csv'
possession_data_url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/Horsens/Horsens_possession_data.csv'

matchstats_df = pd.read_csv(matchstats_url)
possession_data_df = pd.read_csv(possession_data_url)

# Filter for goals (typeId == 16) and select relevant columns
goals = possession_data_df[possession_data_df['typeId'] == 16]
goals = goals[['team_name', 'playerName', 'label', 'date']]

# List of players' goals
player_goal_counts = (
    goals.groupby(['playerName','label','date'])
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
goal_counts['wins'] = goal_counts['result'] > 0

# Filter for Horsens matches and relevant columns
horsens_results = goal_counts[goal_counts['team_name'] == 'Horsens']
horsens_results = horsens_results[['label', 'date', 'wins']]

# Display the dataframes in Streamlit
matchstats_df = matchstats_df[['player_matchName','label','date','player_position','minsPlayed']]
# Assuming matchstats_df is your dataframe

matchstats_df['Subbed_in'] = (matchstats_df['minsPlayed'] > 0) & (matchstats_df['player_position'] == 'Substitute')

# Aggregate the data including Subbed_in
aggregated_df = (
    matchstats_df.groupby(['player_matchName', 'label', 'date'])
    .agg(
        In_squad=('player_matchName', 'count'),  # Count appearances of each player
        Starting_11=('player_position', lambda x: (x != 'Substitute').sum()),  # Count non-substitute entries
        total_minutes_played=('minsPlayed', 'sum'),
        Subbed_in=('Subbed_in', 'sum')  # Sum the Subbed_in column to count occurrences
    )
).reset_index()

aggregated_df = aggregated_df.rename(columns={'player_matchName': 'playerName'})

# Set the index as player_matchName (this is already the case after groupby)
merged_df = aggregated_df.merge(player_goal_counts, on=['playerName', 'label', 'date'],how = 'left')
merged_df = merged_df.merge(horsens_results,on=['label', 'date'],how = 'left')
merged_df['Starting_11_wins'] = merged_df.apply(
    lambda row: 1 if row['Starting_11'] > 0 and row['wins'] == True else 0, axis=1
)
merged_df['Subbed_in_wins'] = merged_df.apply(
    lambda row: 1 if row['Subbed_in'] > 0 and row['wins'] == True else 0, axis=1
)
merged_df = merged_df.sort_values(by=['playerName','date'],ascending=True)
st.dataframe(merged_df,hide_index=True)
# Print the filtered dataframe
final_df = (
    merged_df.groupby('playerName')
    .agg(
        In_squad=('In_squad', 'sum'),
        Starting_11=('Starting_11', 'sum'),
        Subbed_in = ('Subbed_in','sum'),
        total_minutes_played=('total_minutes_played', 'sum'),
        goal_count=('goal_count', 'sum'),
        Starting_11_wins=('Starting_11_wins', 'sum'),
        Subbed_in_wins = ('Subbed_in_wins','sum')
    )
    .reset_index()
)
final_df = final_df.set_index('playerName')

# Display the resulting dataframe
st.dataframe(final_df)
