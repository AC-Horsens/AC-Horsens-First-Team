import pandas as pd
import streamlit as st

st.set_page_config(layout='wide')
url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/Bonus.csv'
matchstats_df = pd.read_csv(url)


matchstats_df = matchstats_df[['player_matchName','player_position','minsPlayed']]
# Assuming matchstats_df is your dataframe
aggregated_df = (
    matchstats_df.groupby('player_matchName')
    .agg(
        In_squad=('player_matchName', 'count'),  # Count appearances of each player
        Starting_11=('player_position', lambda x: (x != 'Substitute').sum()),  # Count non-substitute entries
        total_minutes_played=('minsPlayed', 'sum')  # Sum of minsPlayed
    )
)

# Set the index as player_matchName (this is already the case after groupby)
aggregated_df.index.name = 'player_matchName'

st.dataframe(aggregated_df)