import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import ast
from collections import Counter
from scipy.ndimage import gaussian_filter
from datetime import datetime
from mplsoccer import Pitch, VerticalPitch
from datetime import datetime, timedelta

st.set_page_config(layout='wide')


@st.cache_data
def load_subs():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/subs%20DNK_1_Division_2024_2025.csv'
    df_subs = pd.read_csv(url)
    df_subs['label'] = (df_subs['label'] + ' ' + df_subs['date'])
    return df_subs   

@st.cache_data
def load_match_stats():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/matchstats_all%20DNK_1_Division_2024_2025.csv'
    match_stats = pd.read_csv(url)
    match_stats['label'] = (match_stats['label'] + ' ' + match_stats['date'])
    match_stats['team_name'] = match_stats['team_name'].str.replace(' ','_')
    return match_stats

df_match_stats = load_match_stats()

chosen_team = st.selectbox('Choose team', sorted(df_match_stats['team_name'].unique()))


def load_possession_data():
    url = f'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/{chosen_team}/{chosen_team}_possession_data.csv'
    df_possession = pd.read_csv(url)
    #df_possession = pd.read_csv(r'C:\Users\Seamus-admin\Documents\GitHub\AC-Horsens-First-Team\DNK_1_Division_2024_2025\Horsens\Horsens_possession_data.csv')
    df_possession['label'] = (df_possession['label'] + ' ' + df_possession['date']).astype(str)
    df_possession['team_name'] = df_possession['team_name'].str.replace(' ','_')

    return df_possession

@st.cache_data
def load_def_line_data():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/Horsens_Defensive_line_data.csv'
    def_line = pd.read_csv(url)
    return def_line

@st.cache_data
def load_possession_stats():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/possession_stats_all%20DNK_1_Division_2024_2025.csv'
    df_possession_stats = pd.read_csv(url)
    df_possession_stats['label'] = (df_possession_stats['label'] + ' ' + df_possession_stats['date'])
    return df_possession_stats

@st.cache_data
def load_xg():
    url = f'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/{chosen_team}/{chosen_team}_xg_data.csv'
    df_xg = pd.read_csv(url)
    df_xg['label'] = (df_xg['label'] + ' ' + df_xg['date'])
    df_xg['team_name'].str.replace(' ', '_')
    df_xg = df_xg[['playerName','label','team_name','x','y','321','periodId','timeMin','timeSec','9','24','25','26']]
    return df_xg

@st.cache_data
def load_all_xg():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/xg_all%20DNK_1_Division_2024_2025.csv'
    df_xg_all = pd.read_csv(url)
    df_xg_all['label'] = (df_xg_all['label'] + ' ' + df_xg_all['date'])
    df_xg_all['team_name'].str.replace(' ', '_')
    return df_xg_all

@st.cache_data
def load_pv():
    url = f'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/{chosen_team}/{chosen_team}_pv_data.csv'
    df_pv = pd.read_csv(url)
    df_pv['label'] = (df_pv['label'] + ' ' + df_pv['date'])
    df_pv['id'] = df_pv['id'].astype(str)
    df_pv['team_name'].str.replace(' ', '_')
    return df_pv

@st.cache_data
def load_xA():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/xA_all%20DNK_1_Division_2024_2025.csv'
    df_xA = pd.read_csv(url)
    df_xA['label'] = (df_xA['label'] + ' ' + df_xA['date']).astype(str)
    return df_xA

@st.cache_data
def load_pv_all():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/pv_all%20DNK_1_Division_2024_2025.csv'
    df_pv_all = pd.read_csv(url)
    df_pv_all['label'] = (df_pv_all['label'] + ' ' + df_pv_all['date']).astype(str)
    return df_pv_all

@st.cache_data
def load_squads():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/squads%20DNK_1_Division_2024_2025.csv'
    squads = pd.read_csv(url)
    return squads

@st.cache_data
def load_physical_data():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/Physical%20data_all.csv'
    physical_data = pd.read_csv(url)
    return physical_data

@st.cache_data
def load_set_piece_data():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2024_2025/set_piece_DNK_1_Division_2024_2025.csv'
    df_set_piece = pd.read_csv(url)
    df_set_piece['label'] = (df_set_piece['label'] + ' ' + df_set_piece['date']).astype(str)
    return df_set_piece

df_xA = load_xA()
df_pv = load_pv_all()
df_match_stats = load_match_stats()
df_xg_all = load_all_xg()
squads = load_squads()



def Dashboard():
    df_possession = load_possession_data()
    df_possession['team_name'] = df_possession['team_name'].apply(lambda x: x if x == chosen_team else 'Opponent')
    df_possession['match_state'] = df_possession['match_state'].apply(
        lambda x: x if x == chosen_team or x == 'draw' else 'Opponent'
    )
    st.title(f'{chosen_team} Dashboard')
    df_possession['date'] = pd.to_datetime(df_possession['date'])

    df_possession = df_possession.sort_values(by='date',ascending=False)
    select_all = st.checkbox("Use all matches", value=True)

    all_labels = df_possession['label'].unique()

    if select_all:
        match_choice = all_labels  # All labels selected automatically
    else:
        match_choice = st.multiselect(
            "Choose Match Labels",
            options=all_labels,
            default=[]
        )
    st.write('Choose match state')
    col1,col2,col3 = st.columns(3)
    with col1:
        option1 = st.checkbox(str(chosen_team))
    with col2:
        option2 = st.checkbox('Draw')
    with col3:
        option3 = st.checkbox('Opponent')


    df_possession = df_possession[df_possession['label'].isin(match_choice)]
    df_possession = df_possession.sort_values(by=['date','timeMin', 'timeSec'])  # Sort the events by time
    df_possession['timeMin'] = df_possession['timeMin'].astype(float)
    
        # Initialize list to store game state durations across matches
    game_state_durations = []

    # Track previous state and previous time
    previous_state = None
    previous_time = None
    previous_label = None

    # Iterate through each row
    for index, row in df_possession.iterrows():
        current_label = row['label']
        current_time = row['timeMin']
        current_state = row['match_state']
        
        # If it's the first row, initialize
        if previous_state is None:
            previous_state = current_state
            previous_time = current_time
            previous_label = current_label
            continue

        # If match state changed
        if current_state != previous_state or current_label != previous_label:
            # Determine end_time
            if current_label == previous_label:
                end_time = current_time
            else:
                # Different match → get end time of previous match
                end_time = df_possession[df_possession['label'] == previous_label]['timeMin'].max()

            # Save the previous state
            duration = end_time - previous_time
            game_state_durations.append((previous_label, previous_state, previous_time, end_time, duration))
            
            # Update previous trackers
            previous_state = current_state
            previous_time = current_time
            previous_label = current_label

    # After loop, save the last state
    if previous_state is not None:
        end_time = df_possession[df_possession['label'] == previous_label]['timeMin'].max()
        duration = end_time - previous_time
        game_state_durations.append((previous_label, previous_state, previous_time, end_time, duration))

    # Create DataFrame
    game_state_df = pd.DataFrame(
        game_state_durations,
        columns=['label', 'match_state', 'start_time', 'end_time', 'duration']
    )

    # Optional: remove duplicates if needed
    game_state_df = game_state_df.drop_duplicates()
    if option1 and option2 and option3:
        game_state_df = game_state_df[game_state_df['match_state'].isin([chosen_team, 'draw', 'Opponent'])]
    # Case when two options are selected
    elif option1 and option2:
        game_state_df= game_state_df[game_state_df['match_state'].isin([chosen_team, 'draw'])]
    elif option1 and option3:
        game_state_df = game_state_df[game_state_df['match_state'].isin([chosen_team, 'Opponent'])]
    elif option2 and option3:
        game_state_df = game_state_df[game_state_df['match_state'].isin(['draw', 'Opponent'])]
    # Case when only one option is selected
    elif option1:
        game_state_df = game_state_df[game_state_df['match_state'] == chosen_team]
    elif option2:
        game_state_df = game_state_df[game_state_df['match_state'] == 'draw']
    elif option3:
        game_state_df = game_state_df[game_state_df['match_state'] == 'Opponent']
    # Show
    game_state_df = game_state_df.groupby('match_state')['duration'].sum().reset_index()
    st.dataframe(game_state_df,hide_index=True)
    # Calculate passes per possession
    state_duration = game_state_df['duration'].sum()
    if option1 and option2 and option3:
        df_possession = df_possession[df_possession['match_state'].isin([chosen_team, 'draw', 'Opponent'])]
    # Case when two options are selected
    elif option1 and option2:
        df_possession = df_possession[df_possession['match_state'].isin([chosen_team, 'draw'])]
    elif option1 and option3:
        df_possession = df_possession[df_possession['match_state'].isin([chosen_team, 'Opponent'])]
    elif option2 and option3:
        df_possession = df_possession[df_possession['match_state'].isin(['draw', 'Opponent'])]
    # Case when only one option is selected
    elif option1:
        df_possession = df_possession[df_possession['match_state'] == chosen_team]
    elif option2:
        df_possession = df_possession[df_possession['match_state'] == 'draw']
    elif option3:
        df_possession = df_possession[df_possession['match_state'] == 'Opponent']


    Pass_per_possession = df_possession[df_possession['typeId'] == 1].groupby(['possessionId', 'label', 'team_name']).size().reset_index(name='Passes per possession')
    Pass_per_possession = Pass_per_possession.drop(columns=['possessionId', 'label'])
    Pass_per_possession = Pass_per_possession.groupby('team_name').mean().reset_index()
    # Calculate xG per match per team
    xg_per_match = df_possession[df_possession['321.0'] > 0]
    xg_per_match = xg_per_match[['team_name', 'label', '321.0']]
    xg_per_match = xg_per_match.groupby(['team_name', 'label']).sum().reset_index()

    cleaned_xg_per_match = df_possession[df_possession['321.0'] > 0.1]
    cleaned_xg_per_match = cleaned_xg_per_match[['team_name', 'label', '321.0']]
    cleaned_xg_per_match = cleaned_xg_per_match.groupby(['team_name', 'label']).sum().reset_index()

    # Ensure both 'Horsens' and 'Opponent' exist for every match
    all_labels = df_possession['label'].unique()
    teams = [chosen_team, 'Opponent']
    full_index = pd.MultiIndex.from_product([teams, all_labels], names=['team_name', 'label'])
    xg_per_match = xg_per_match.set_index(['team_name', 'label']).reindex(full_index, fill_value=0).reset_index()
    cleaned_xg_per_match = cleaned_xg_per_match.set_index(['team_name', 'label']).reindex(full_index, fill_value=0).reset_index()

    # Calculate total xG per match
    total_xg_per_match = xg_per_match.groupby('label')['321.0'].sum().reset_index()
    total_xg_per_match = total_xg_per_match.rename(columns={'321.0': 'total_match_xG'})
    cleaned_total_xg_per_match = cleaned_xg_per_match.groupby('label')['321.0'].sum().reset_index()
    cleaned_total_xg_per_match = cleaned_total_xg_per_match.rename(columns={'321.0': 'total_match_cleaned_xG'})

    # Merge team xG with total match xG (on 'label')
    xg_per_match = xg_per_match.merge(total_xg_per_match, on='label', how='left')
    cleaned_xg_per_match = cleaned_xg_per_match.merge(cleaned_total_xg_per_match, on='label', how='left')
    # Calculate xG difference and xG against
    xg_per_match['xG_diff'] = 2 * xg_per_match['321.0'] - xg_per_match['total_match_xG']
    xg_per_match['xG against'] = xg_per_match['total_match_xG'] - xg_per_match['321.0']

    cleaned_xg_per_match['Cleaned xG difference'] = 2 * cleaned_xg_per_match['321.0'] - cleaned_xg_per_match['total_match_cleaned_xG']
    cleaned_xg_per_match['Cleaned xG against'] = cleaned_xg_per_match['total_match_cleaned_xG'] - cleaned_xg_per_match['321.0']

    # Now average xG and xG_diff per team
    xg_summary = xg_per_match.groupby('team_name').agg({
        '321.0': 'sum',
        'xG_diff': 'sum',
        'xG against': 'sum'       # xG against
    }).reset_index()
    
    cleaned_xg_summary = cleaned_xg_per_match.groupby('team_name').agg({
        '321.0': 'sum',
        'Cleaned xG difference': 'sum',
        'Cleaned xG against': 'sum'       # xG against
    }).reset_index()
    cleaned_xg_summary = cleaned_xg_summary.rename(columns={'321.0': 'Cleaned xG'})

    xg_summary = xg_summary.rename(columns={'321.0': 'xG', 'xG_diff': 'xG difference'})
    xg_summary = xg_summary.merge(cleaned_xg_summary, on ='team_name',how='outer')
    team_summary = xg_summary.merge(Pass_per_possession, on='team_name',how='outer')
    team_summary['xG per 90'] = (team_summary['xG']/state_duration)*90
    team_summary['xG difference per 90'] = (team_summary['xG difference']/state_duration)*90
    team_summary['xG against per 90'] = (team_summary['xG against']/state_duration)*90
    team_summary['Cleaned xG per 90'] = (team_summary['Cleaned xG'] / state_duration) * 90
    team_summary['Cleaned xG difference per 90'] = (team_summary['Cleaned xG difference'] / state_duration) * 90
    team_summary['Cleaned xG against per 90'] = (team_summary['Cleaned xG against'] / state_duration) * 90

    team_summary = team_summary.round(2)
    metrics_df = team_summary[['team_name', 'xG', 'xG difference', 'xG against', 'Cleaned xG', 'Cleaned xG difference', 'Cleaned xG against', 'Passes per possession']]
    per90_df = team_summary[['team_name', 
                            'xG per 90', 
                            'xG difference per 90', 
                            'xG against per 90',
                            'Cleaned xG per 90', 
                            'Cleaned xG difference per 90', 
                            'Cleaned xG against per 90',
                            'Passes per possession']]

    # Display one above the other
    st.subheader("Team Summary (Total Metrics)")
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    st.subheader("Team Summary (Per 90 Minutes)")
    st.dataframe(per90_df, hide_index=True, use_container_width=True)

    def team_mentality_score():
        df_opponent = df_possession[
            (df_possession['team_name'] == 'Opponent') & 
            (df_possession['x'] > 75) & 
            (df_possession['typeId'].isin([1, 2, 3, 13, 14, 15, 16]))
        ]
        # Count the number of actions (rows) where the x_axis condition is met
        actions_count = len(df_opponent)

        # Filter again for rows where column '321.0' is greater than 0.15
        df_opponent_321 = df_opponent[
            ((df_opponent['321.0'] > 0.15) | (df_opponent['318.0'] > 0.15)| (df_opponent['322.0'] > 0.15)) |
            ((df_opponent['x'] > 83) & (df_opponent['y'] > 21.1) & (df_opponent['y'] < 78.9))
        ]
        df_opponent['weight'] = (
            (df_opponent['321.0'] > 0.15).astype(int) * 1 +
            (df_opponent['318.0'] > 0.15).astype(int) * 1 +
            (df_opponent['322.0'] > 0.15).astype(int) * 1 +
            ((df_opponent['x'] > 83) & (df_opponent['y'].between(21.1, 78.9))).astype(int) * 0.5
        )
        
        weighted_actions_sum = df_opponent['weight'].sum()
        
        mentality_score = (1 - (weighted_actions_sum / actions_count)) * 100
        mentality_score = mentality_score.round(2)
        actions_321_count = len(df_opponent_321)
        mentality_scores = []

        # Find all unique match labels in the *full* df_possession (not filtered)
        all_labels = df_possession['label'].unique()

        # Loop through all matches
        for label in all_labels:
            df_match = df_possession[df_possession['label'] == label]
            df_opponent = df_match[
                (df_match['team_name'] == 'Opponent') & 
                (df_match['x'] > 75) & 
                (df_match['typeId'].isin([1, 2, 3, 13, 14, 15, 16]))
            ]
            actions_count = len(df_opponent)
            
            if actions_count > 0:
                # Calculate weighted sum
                df_opponent['weight'] = (
                    (df_opponent['321.0'] > 0.15).astype(int) * 1 +
                    (df_opponent['318.0'] > 0.15).astype(int) * 1 +
                    (df_opponent['322.0'] > 0.15).astype(int) * 1 +
                    ((df_opponent['x'] > 83) & (df_opponent['y'].between(21.1, 78.9))).astype(int) * 0.5
                )
                
                weighted_actions_sum = df_opponent['weight'].sum()
                
                mentality_score = (1 - (weighted_actions_sum / actions_count)) * 100
            else:
                mentality_score = None

            mentality_scores.append({'Match': label, 'Team Mentality Score': mentality_score})

        # Convert list to dataframe
        mentality_df = pd.DataFrame(mentality_scores)

        # Show the mentality scores

        fig = px.line(
            mentality_df,
            x='Match', 
            y='Team Mentality Score',
            title='Team Mentality Score per Match',
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis_range=[50, 100],  # Set y-axis limits
        )

        st.plotly_chart(fig, use_container_width=True)

    def defensive_line_data():
        def_line = load_def_line_data()
        labels_df = df_possession[['match_id','date', 'label']].drop_duplicates()
        states_df = df_possession[['match_id','date','label', 'contestantId', 'timeMin', 'timeSec', 'match_state']]

        # Merge only on match_id to get label
        def_line = def_line.merge(labels_df, on='match_id', how='left')

        # Merge on full key to get match_state
        def_line = def_line.merge(states_df, on=['match_id','date','label', 'contestantId', 'timeMin', 'timeSec'], how='left')
        def_line = def_line.sort_values(['date','timeMin','timeSec'])
        def_line = def_line.fillna(method='ffill')


        def_line = def_line[['match_id','label','team_name','date','contestantId','timeMin','timeSec','percent_succes','match_state']]
        def_line = def_line.sort_values(['date','timeMin','timeSec'])

        def_line = def_line.groupby(['label','date'])['percent_succes'].mean().reset_index()
        def_line = def_line.sort_values(['date'])

        fig = px.line(
            def_line,
            x='label', 
            y='percent_succes',
            title='Defensive line succesrate',
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis_range=[85, 100],  # Set y-axis limits
        )

        st.plotly_chart(fig, use_container_width=True)

    def set_pieces():
        df_set_pieces = load_set_piece_data()
        df_set_pieces = df_set_pieces.fillna(0)
        df_set_pieces = df_set_pieces.round(2)
        df_set_pieces_goals = df_set_pieces[df_set_pieces['typeId'] == 16]
        df_set_pieces_goals = df_set_pieces_goals.groupby('team_name').size().reset_index(name='Goals')
        df_set_pieces_goals = df_set_pieces_goals.sort_values(by='Goals',ascending=False)
        df_set_pieces_matches = df_set_pieces.groupby(['team_name','label']).agg({'321.0':'sum'}).reset_index()
        df_set_pieces_matches['xG_match'] = df_set_pieces_matches.groupby('label')['321.0'].transform('sum')
        df_set_pieces_matches['xG_against'] = df_set_pieces_matches['321.0'] - df_set_pieces_matches['xG_match']
        df_set_pieces_matches['xG_diff'] = df_set_pieces_matches['321.0'] - df_set_pieces_matches['xG_match'] + df_set_pieces_matches['321.0']

        df_set_pieces_sum = df_set_pieces_matches.groupby('team_name').agg({'321.0': 'sum', 'xG_against': 'sum', 'xG_diff': 'sum'})
        df_set_pieces_sum = df_set_pieces_sum.rename(columns={'321.0': 'xG'})
        df_set_pieces_sum = df_set_pieces_sum.sort_values(by='xG',ascending=False)
        st.header('Whole season')
        st.write('All set pieces')
        st.dataframe(df_set_pieces_goals,hide_index=True)
        st.dataframe(df_set_pieces_sum)
        st.write('Freekicks')
        Freekicks = df_set_pieces[(df_set_pieces['set_piece_type'] == 'freekick')]
        Freekicks = Freekicks.groupby(['team_name','label']).agg({'321.0':'sum'}).reset_index()
        Freekicks['xG_match'] = Freekicks.groupby('label')['321.0'].transform('sum')
        Freekicks['xG_against'] = Freekicks['321.0'] - Freekicks['xG_match']
        Freekicks['xG_diff'] = Freekicks['321.0'] - Freekicks['xG_match'] + Freekicks['321.0']
        Freekicks = Freekicks.groupby('team_name').agg({'321.0': 'sum', 'xG_against': 'sum', 'xG_diff': 'sum'})
        Freekicks = Freekicks.rename(columns={'321.0': 'xG'})
        Freekicks = Freekicks.sort_values(by='xG',ascending=False)
        st.dataframe(Freekicks)

        st.write('Corners')
        Corners = df_set_pieces[df_set_pieces['set_piece_type'] =='corner']
        #Corners = df_set_pieces[(df_set_pieces['26.0'] != True) & (df_set_pieces['24.0'] != True)]
        Corners = Corners.groupby(['team_name','label']).agg({'321.0':'sum'}).reset_index()
        Corners['xG_match'] = Corners.groupby('label')['321.0'].transform('sum')
        Corners['xG_against'] = Corners['321.0'] - Corners['xG_match']
        Corners['xG_diff'] = Corners['321.0'] - Corners['xG_match'] + Corners['321.0']
        Corners = Corners.groupby('team_name').agg({'321.0': 'sum', 'xG_against': 'sum', 'xG_diff': 'sum'})

        Corners = Corners.rename(columns={'321.0': 'xG'})
        Corners = Corners.sort_values(by='xG',ascending=False)
        st.dataframe(Corners)
        
        st.write('Throw ins')
        Throw_ins = df_set_pieces[df_set_pieces['set_piece_type'] =='throw_in']
        #Corners = df_set_pieces[(df_set_pieces['26.0'] != True) & (df_set_pieces['24.0'] != True)]
        Throw_ins = Throw_ins.groupby(['team_name','label']).agg({'321.0':'sum'}).reset_index()
        Throw_ins['xG_match'] = Throw_ins.groupby('label')['321.0'].transform('sum')
        Throw_ins['xG_against'] = Throw_ins['321.0'] - Throw_ins['xG_match']
        Throw_ins['xG_diff'] = Throw_ins['321.0'] - Throw_ins['xG_match'] + Throw_ins['321.0']
        Throw_ins = Throw_ins.groupby('team_name').agg({'321.0': 'sum', 'xG_against': 'sum', 'xG_diff': 'sum'})

        Throw_ins = Throw_ins.rename(columns={'321.0': 'xG'})
        Throw_ins = Throw_ins.sort_values(by='xG',ascending=False)
        st.dataframe(Throw_ins)

        st.header('Chosen matches')
        df_set_pieces_matches1 = df_set_pieces[df_set_pieces['label'].isin(match_choice)]
        df_set_pieces_matches = df_set_pieces_matches1.groupby(['team_name','label']).agg({'321.0':'sum'}).reset_index()
        df_set_pieces_matches['xG_match'] = df_set_pieces_matches.groupby('label')['321.0'].transform('sum')
        df_set_pieces_matches['xG_against'] = df_set_pieces_matches['321.0'] - df_set_pieces_matches['xG_match']
        df_set_pieces_matches['xG_diff'] = df_set_pieces_matches['321.0'] - df_set_pieces_matches['xG_match'] + df_set_pieces_matches['321.0']
        df_set_pieces_matches['team_name'] = df_set_pieces_matches['team_name'].apply(lambda x: 'Opponent' if x != 'Horsens' else x)
        df_set_pieces_sum = df_set_pieces_matches.groupby('team_name').agg({'321.0': 'sum', 'xG_against': 'sum', 'xG_diff': 'sum'})
        df_set_pieces_sum = df_set_pieces_sum.rename(columns={'321.0': 'xG'})
        df_set_pieces_sum = df_set_pieces_sum.sort_values(by='xG',ascending=False)

        df_set_pieces_matches = df_set_pieces_matches[['team_name','321.0','xG_against','xG_diff']]
        df_set_pieces_matches = df_set_pieces_matches.rename(columns={'321.0': 'xG'})
        st.write('All set pieces')
        st.dataframe(df_set_pieces_sum)

        st.write('Freekicks')
        Freekicks = df_set_pieces_matches1[(df_set_pieces_matches1['set_piece_type'] == 'freekick')]
        Freekicks['team_name'] = Freekicks['team_name'].apply(lambda x: 'Opponent' if x != 'Horsens' else x)
        Freekicks = Freekicks.groupby(['team_name','label']).agg({'321.0':'sum'}).reset_index()
        Freekicks['xG_match'] = Freekicks.groupby('label')['321.0'].transform('sum')
        Freekicks['xG_against'] = Freekicks['321.0'] - Freekicks['xG_match']
        Freekicks['xG_diff'] = Freekicks['321.0'] - Freekicks['xG_match'] + Freekicks['321.0']
        Freekicks = Freekicks.rename(columns={'321.0': 'xG'})

        Freekicks = Freekicks.sort_values(by='xG',ascending=False)
        Freekicks_matches = Freekicks[['team_name','xG','xG_against','xG_diff']]
        Freekicks_matches = Freekicks_matches.groupby('team_name').sum()
        st.dataframe(Freekicks_matches)

        st.write('Corners')
        Corners = df_set_pieces_matches1[(df_set_pieces_matches1['set_piece_type'] == 'corner')]
        Corners['team_name'] = Corners['team_name'].apply(lambda x: 'Opponent' if x != 'Horsens' else x)
        Corners = Corners.groupby(['team_name','label']).agg({'321.0':'sum'}).reset_index()
        Corners['xG_match'] = Corners.groupby('label')['321.0'].transform('sum')
        Corners['xG_against'] = Corners['321.0'] - Corners['xG_match']
        Corners['xG_diff'] = Corners['321.0'] - Corners['xG_match'] + Corners['321.0']
        Corners = Corners.rename(columns={'321.0': 'xG'})

        Corners = Corners.sort_values(by='xG',ascending=False)
        Corners_matches = Corners[['team_name','xG','xG_against','xG_diff']]
        Corners_matches = Corners_matches.groupby('team_name').sum()

        st.dataframe(Corners_matches)

        st.write('Throw ins')
        Throw_ins = df_set_pieces_matches1[df_set_pieces_matches1['set_piece_type'] =='throw_in']
        #Corners = df_set_pieces[(df_set_pieces['26.0'] != True) & (df_set_pieces['24.0'] != True)]
        Throw_ins = Throw_ins.groupby(['team_name','label']).agg({'321.0':'sum'}).reset_index()
        Throw_ins['xG_match'] = Throw_ins.groupby('label')['321.0'].transform('sum')
        Throw_ins['xG_against'] = Throw_ins['321.0'] - Throw_ins['xG_match']
        Throw_ins['xG_diff'] = Throw_ins['321.0'] - Throw_ins['xG_match'] + Throw_ins['321.0']
        Throw_ins = Throw_ins.groupby('team_name').agg({'321.0': 'sum', 'xG_against': 'sum', 'xG_diff': 'sum'})

        Throw_ins = Throw_ins.rename(columns={'321.0': 'xG'})
        Throw_ins = Throw_ins.sort_values(by='xG',ascending=False)
        st.dataframe(Throw_ins)

    Data_types = {
        'Team mentality score': team_mentality_score,
        'Defensive line': defensive_line_data,
        'Set pieces': set_pieces
    }

    for i in range(1, 4):
        if f'selected_data{i}' not in st.session_state:
            st.session_state[f'selected_data{i}'] = ''


    # Create three columns for select boxes
    col1, col2, col3 = st.columns(3)

    # Function to create selectbox and update session state without rerunning entire page
    def create_selectbox(column, key):
        with column:
            selected_data = st.selectbox(f'Choose data type {key[-1]}', [''] + list(Data_types.keys()), key=key)
            if selected_data and selected_data != st.session_state[key]:
                st.session_state[key] = selected_data
                st.experimental_rerun()

    # Create select boxes for each column
    create_selectbox(col1, 'selected_data1')
    create_selectbox(col2, 'selected_data2')
    create_selectbox(col3, 'selected_data3')

    # Display the current selection results in columns
    with col1:
        if st.session_state['selected_data1']:
            Data_types[st.session_state['selected_data1']]()

    with col2:
        if st.session_state['selected_data2']:
            Data_types[st.session_state['selected_data2']]()

    with col3:
        if st.session_state['selected_data3']:
            Data_types[st.session_state['selected_data3']]()

Data_types = {
    'Dashboard': Dashboard,
}

Dashboard()
st.cache_data()
st.cache_resource()

st.cache_data()
st.cache_resource()

