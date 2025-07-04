import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
from datetime import datetime
from datetime import date


def load_data():
    df_xg = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/xg_all DNK_1_Division_2024_2025.csv')
    df_xg['label'] = df_xg['label'] + ' ' + df_xg['date']

    df_xa = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/xA_all DNK_1_Division_2024_2025.csv')
    df_xa['label'] = df_xa['label'] + ' ' + df_xa['date']

    df_pv = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/pv_all DNK_1_Division_2024_2025.csv')
    df_pv['label'] = df_pv['label'] + ' ' + df_pv['date']

    df_possession_stats = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/possession_stats_all DNK_1_Division_2024_2025.csv')
    df_possession_stats['label'] = df_possession_stats['label'] + ' ' + df_possession_stats['date']

    df_xa_agg = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/Horsens/Horsens_possession_data.csv')
    df_xa_agg['label'] = df_xa_agg['label'] + ' ' + df_xa_agg['date']

    df_possession_data = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/Horsens/Horsens_possession_data.csv')
    df_possession_data['label'] = df_possession_data['label'] + ' ' + df_possession_data['date']

    df_xg_agg = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/Horsens/Horsens_xg_data.csv')
    df_xg_agg['label'] = df_xg_agg['label'] + ' ' + df_xg_agg['date']

    df_pv_agg = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/Horsens/Horsens_pv_data.csv')
    df_pv_agg['label'] = df_pv_agg['label'] + ' ' + df_pv_agg['date']

    df_possession_xa = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/xA_all DNK_1_Division_2024_2025.csv')
    df_possession_xa['label'] = df_possession_xa['label'] + ' ' + df_possession_xa['date']

    df_xg_all = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/xg_all DNK_1_Division_2024_2025.csv')
    df_xg_all['label'] = df_xg_all['label'] + ' ' + df_xg_all['date']

    df_pv_all = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/pv_all DNK_1_Division_2024_2025.csv')
    df_pv_all['label'] = df_pv_all['label'] + ' ' + df_pv_all['date']

    df_matchstats = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/matchstats_all DNK_1_Division_2024_2025.csv')
    df_matchstats['label'] = df_matchstats['label'] + ' ' + df_matchstats['date']

    squads = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/squads DNK_1_Division_2024_2025.csv')
    
    possession_events = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/Horsens/Horsens_possession_data.csv')
    possession_events['label'] = possession_events['label'] + ' ' + possession_events['date']
    #packing_df = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/packing_all DNK_1_Division_2024_2025.csv')
    #packing_df['label'] = packing_df['label'] + ' ' + packing_df['date']
    
    #space_control_df = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2024_2025/Space_control_all DNK_1_Division_2024_2025.csv')
    #space_control_df['label'] = space_control_df['label'] + ' ' + space_control_df['date']
    
    return df_xg, df_xa, df_pv, df_possession_stats, df_xa_agg, df_possession_data, df_xg_agg, df_pv_agg, df_xg_all, df_possession_xa, df_pv_all, df_matchstats, squads, possession_events

def create_stacked_bar_chart(win_prob, draw_prob, loss_prob, title, filename):
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Define the colors for each segment
    colors = ['green', 'yellow', 'red']
    segments = [win_prob, draw_prob, loss_prob]
    labels = ['Win', 'Draw', 'Loss']
    
    # Plot the stacked bar segments
    left = 0
    for seg, color, label in zip(segments, colors, labels):
        ax.barh(0, seg, left=left, color=color, height=0.5, label=label)
        left += seg
    
    # Add text annotations for each segment
    left = 0
    for seg, color, label in zip(segments, colors, labels):
        ax.text(left + seg / 2, 0, f"{label}: {seg:.2f}", ha='center', va='center', fontsize=10, color='black', fontweight='bold')
        left += seg
    
    # Formatting
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')
    plt.title(label=title, fontsize=12, fontweight='bold', y=1.2, va='top', loc='left')
    plt.savefig(filename, bbox_inches='tight', dpi=500)  # Use high DPI for better quality
    plt.close()

def create_bar_chart(value, title, filename, max_value, thresholds, annotations):
    fig, ax = plt.subplots(figsize=(8, 2))
    
    if value < thresholds[0]:
        bar_color = 'red'
    elif value < thresholds[1]:
        bar_color = 'orange'
    elif value < thresholds[2]:
        bar_color = 'yellow'
    else:
        bar_color = 'green'
    
    # Plot the full bar background
    ax.barh(0, max_value, color='lightgrey', height=0.5)
    
    # Plot the value bar with the determined color
    ax.barh(0, value, color=bar_color, height=0.5)
    # Plot the thresholds with annotations
    for threshold, color, annotation in zip(thresholds, ['red', 'yellow', 'green'], annotations):
        ax.axvline(threshold, color=color, linestyle='--', linewidth=1.5)
        ax.text(threshold, 0.3, f"{threshold:.2f}", ha='center', va='center', fontsize=10, color=color)
        ax.text(threshold, -0.3, annotation, ha='center', va='center', fontsize=10, color=color)
    
    # Add the text for title and value
    ax.text(value, -0.0, f"{value:.2f}", ha='center', va='center', fontsize=12, fontweight='bold', color='black')
        
    # Formatting
    ax.set_xlim(0, max_value)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')
    plt.title(label=title, fontsize=12, fontweight='bold',y=1.2,va='top', loc='left')   
    plt.savefig(filename, bbox_inches='tight', dpi=500)  # Use high DPI for better quality
    plt.close()

def generate_cumulative_chart(df, column, title, filename):
    plt.figure(figsize=(12, 6))
    for team in df['team_name'].unique():
        team_data = df[df['team_name'] == team]
        plt.plot(team_data['timeMin'] + team_data['timeSec'] / 60, team_data[column], label=team)
    
    plt.xlabel('Time (minutes)')
    plt.title(f'Cumulative {title}')
    plt.legend()
    plt.savefig(filename, bbox_inches='tight', dpi=500)
    plt.close()

def generate_possession_chart(df_possession_stats, filename):
    cols_to_average = df_possession_stats.columns[[6, 7, 8]]
    df_possession_stats[cols_to_average] = df_possession_stats[cols_to_average].apply(pd.to_numeric, errors='coerce')
    time_column = 'interval'

    # Calculate sliding average
    sliding_average = df_possession_stats[cols_to_average].rolling(window=2).mean()
    # Plotting
    plt.figure(figsize=(12,6))
    sorted_columns = sliding_average.columns.sort_values()  # Sort the columns alphabetically
    for col in sorted_columns:
        plt.plot(df_possession_stats[time_column], sliding_average[col], label=col)

    plt.xlabel('Time (minutes)')
    plt.title('Sliding average territorial possession')
    plt.legend(loc='upper left')
    plt.savefig(filename, bbox_inches='tight', dpi=500)
    plt.close()

def sliding_average_plot(df, window_size=3, filename=None):
    # Sort the DataFrame by 'date'
    df_sorted = df.sort_values(by='date')
    df_sorted = df_sorted.reset_index()
    df_sorted['label'] = (df_sorted.index + 1).astype(str) + '. ' + df_sorted['label'].str[:-10]

    # Calculate the sliding average and cumulative average
    sliding_average = df_sorted['expected_points'].rolling(window=window_size).mean()
    cumulative_average = df_sorted['expected_points'].expanding().mean()

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 3))
    # Adjust the subplot to give more room for the x-axis labels
    fig.subplots_adjust(bottom=0.3)
    
    # Plot the sliding average line and cumulative average
    line1, = ax.plot(df_sorted['label'], sliding_average, color='blue', label='3-game rolling average')
    line2, = ax.plot(df_sorted['label'], cumulative_average, color='black', label='Cumulative Average')

    # Add horizontal lines and annotations
    line3 = ax.axhline(y=1, color='red', linestyle='--', label='Relegation')
    line4 = ax.axhline(y=1.3, color='yellow', linestyle='--', label='Top 6')
    line5 = ax.axhline(y=1.8, color='green', linestyle='--', label='Promotion')

    # Set y-axis limits
    ax.set_ylim(0, 3)

    # Add the first legend for sliding and cumulative averages
    legend1 = ax.legend(handles=[line1, line2], loc='upper left', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6)
    ax.add_artist(legend1)  # Add the first legend manually to the axes

    # Add the second legend for horizontal lines
    ax.legend(handles=[line3, line4, line5], loc='upper right', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6)

    # Set labels and title
    ax.set_ylabel('Expected Points',fontsize=6)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=6)

    # Show grid
    ax.grid(True)
    
    # Set the x-axis limits
    ax.set_xlim(df_sorted['label'].iloc[0], df_sorted['label'].iloc[-1])

    ax.axhspan(0, 1, color='red', alpha=0.6)  # Below relegation
    ax.axhspan(1, 1.3, color='orange', alpha=0.6)  # Between relegation and top 6
    ax.axhspan(1.3, 1.8, color='yellow', alpha=0.6)  # Between top 6 and promotion
    ax.axhspan(1.8, 3, color='green', alpha=0.6)  # Above promotion

    # Save the plot to a file if filename is provided
    if filename:
        plt.savefig(filename, format='png', dpi=500, bbox_inches='tight')
    else:
        plt.show()
        
def simulate_goals(values, num_simulations=100000):
    return np.random.binomial(1, values[:, np.newaxis], (len(values), num_simulations)).sum(axis=0)

def simulate_match(home_values, away_values, num_simulations=100000):
    home_goals_simulated = simulate_goals(home_values, num_simulations)
    away_goals_simulated = simulate_goals(away_values, num_simulations)
    
    home_wins = np.sum(home_goals_simulated > away_goals_simulated)
    draws = np.sum(home_goals_simulated == away_goals_simulated)
    away_wins = np.sum(home_goals_simulated < away_goals_simulated)
    
    home_points = (home_wins * 3 + draws) / num_simulations
    away_points = (away_wins * 3 + draws) / num_simulations
    
    home_win_prob = home_wins / num_simulations
    draw_prob = draws / num_simulations
    away_win_prob = away_wins / num_simulations
    
    return home_points, away_points, home_win_prob, draw_prob, away_win_prob

def calculate_expected_points(df, value_column):
    expected_points_list = []
    total_expected_points = {team: 0 for team in df['team_name'].unique()}
    
    matches = df.groupby('label')
    for label, match_df in matches:
        teams = match_df['team_name'].unique()
        if len(teams) == 2:
            home_team, away_team = teams
            home_values = match_df[match_df['team_name'] == home_team][value_column].values
            away_values = match_df[match_df['team_name'] == away_team][value_column].values
            
            home_points, away_points, home_win_prob, draw_prob, away_win_prob = simulate_match(home_values, away_values)
            match_date = match_df['date'].iloc[0]
            
            expected_points_list.append({
                'label': label,
                'date' : match_date, 
                'team_name': home_team, 
                'expected_points': home_points, 
                'win_probability': home_win_prob, 
                'draw_probability': draw_prob, 
                'loss_probability': away_win_prob
            })
            expected_points_list.append({
                'label': label,
                'date' : match_date, 
                'team_name': away_team, 
                'expected_points': away_points, 
                'win_probability': away_win_prob, 
                'draw_probability': draw_prob, 
                'loss_probability': home_win_prob
            })
            
            total_expected_points[home_team] += home_points
            total_expected_points[away_team] += away_points
    
    expected_points_df = pd.DataFrame(expected_points_list)
    total_expected_points_df = pd.DataFrame(list(total_expected_points.items()), columns=['team_name', 'total_expected_points'])
    total_expected_points_df = total_expected_points_df.sort_values(by='total_expected_points', ascending=False)
    return expected_points_df, total_expected_points_df

def calculate_expected_points_top_6(df, value_column):
    target_teams = {'Hvidovre', 'Horsens', 'Fredericia', 'Esbjerg', 'OB', 'Kolding'}
    
    expected_points_list = []
    total_expected_points = {team: 0 for team in df['team_name'].unique()}
    
    matches = df.groupby('label')
    
    for label, match_df in matches:
        teams = match_df['team_name'].unique()
        
        # Check if the match is between two of the target teams
        if len(teams) == 2 and set(teams).issubset(target_teams):
            home_team, away_team = teams
            home_values = match_df[match_df['team_name'] == home_team][value_column].values
            away_values = match_df[match_df['team_name'] == away_team][value_column].values
            
            home_points, away_points, home_win_prob, draw_prob, away_win_prob = simulate_match(home_values, away_values)
            match_date = match_df['date'].iloc[0]
            
            expected_points_list.append({
                'label': label,
                'date': match_date,
                'team_name': home_team,
                'expected_points': home_points,
                'win_probability': home_win_prob,
                'draw_probability': draw_prob,
                'loss_probability': away_win_prob
            })
            expected_points_list.append({
                'label': label,
                'date': match_date,
                'team_name': away_team,
                'expected_points': away_points,
                'win_probability': away_win_prob,
                'draw_probability': draw_prob,
                'loss_probability': home_win_prob
            })
            
            total_expected_points[home_team] += home_points
            total_expected_points[away_team] += away_points
    
    expected_points_df = pd.DataFrame(expected_points_list)
    total_expected_points_df = pd.DataFrame(list(total_expected_points.items()), columns=['team_name', 'total_expected_points'])
    total_expected_points_df = total_expected_points_df.sort_values(by='total_expected_points', ascending=False)
    return expected_points_df, total_expected_points_df

def preprocess_data(df_xg_agg, df_xa_agg, df_pv_agg, df_possession_stats):
    df_xg_agg['timeMin'] = df_xg_agg['timeMin'].astype(int)
    df_xg_agg['timeSec'] = df_xg_agg['timeSec'].astype(int)
    df_xg_agg = df_xg_agg.sort_values(by=['team_name','timeMin', 'timeSec'])
    df_xg_agg['cumulativxg'] = df_xg_agg.groupby(['team_name','label'])['321'].cumsum()

    df_xa_agg['timeMin'] = df_xa_agg['timeMin'].astype(int)
    df_xa_agg['timeSec'] = df_xa_agg['timeSec'].astype(int)
    df_xa_agg = df_xa_agg.sort_values(by=['team_name','timeMin', 'timeSec'])
    df_xa_agg = df_xa_agg[df_xa_agg['318.0'].astype(float) > 0]

    df_xa_agg['cumulativxa'] = df_xa_agg.groupby(['team_name', 'label'])['318.0'].cumsum()
    
    df_possession_stats = df_possession_stats[df_possession_stats['type'] == 'territorialThird'].copy()
    df_possession_stats.loc[:, 'home'] = df_possession_stats['home'].astype(float)
    df_possession_stats.loc[:, 'away'] = df_possession_stats['away'].astype(float)

    df_possession_stats_summary = df_possession_stats.groupby(['home_team', 'away_team', 'label']).agg({'home': 'mean', 'away': 'mean'}).reset_index()
    df_possession_stats_summary = df_possession_stats_summary.rename(columns={'home': 'home_possession', 'away': 'away_possession'})

    df_possession_stats = df_possession_stats.drop_duplicates()
    df_possession_stats = df_possession_stats[df_possession_stats['interval_type'] == 5]

    return df_xg_agg, df_xa_agg, df_pv_agg, df_possession_stats, df_possession_stats_summary

def calculate_ppda(df_possession_data):
    df_ppda = df_possession_data[df_possession_data['typeId'].isin([1, 4, 7,8, 45])]
    df_ppdabeyond40 = df_ppda[df_ppda['x'].astype(float) > 40]
    df_ppdabeyond40_passes = df_ppdabeyond40[df_ppdabeyond40['typeId'] == 1]
    df_ppdabeyond40_passestotal = df_ppdabeyond40_passes.groupby('label')['eventId'].count().reset_index()
    df_ppdabeyond40_passestotal = df_ppdabeyond40_passestotal.rename(columns={'eventId': 'passes in game'})
    df_ppdabeyond40_passesteams = df_ppdabeyond40_passes.groupby(['label','team_name'])['eventId'].count().reset_index()
    df_ppdabeyond40_passesteams = df_ppdabeyond40_passesteams.rename(columns={'eventId': 'passes'})

    df_ppdabeyond40_defactions = df_ppdabeyond40[df_ppdabeyond40['typeId'].isin([4, 7, 8, 45])]
    df_ppdabeyond40_defactionstotal = df_ppdabeyond40_defactions.groupby('label')['eventId'].count().reset_index()
    df_ppdabeyond40_defactionstotal = df_ppdabeyond40_defactionstotal.rename(columns={'eventId': 'defensive actions in game'})
    df_ppdabeyond40_defactionsteams = df_ppdabeyond40_defactions.groupby(['label', 'team_name'])['eventId'].count().reset_index()
    df_ppdabeyond40_defactionsteams = df_ppdabeyond40_defactionsteams.rename(columns={'eventId': 'defensive actions'})
    df_ppdabeyond40total = df_ppdabeyond40_defactionstotal.merge(df_ppdabeyond40_passestotal)
    df_ppdabeyond40 = df_ppdabeyond40_defactionsteams.merge(df_ppdabeyond40total)
    df_ppdabeyond40 = df_ppdabeyond40.merge(df_ppdabeyond40_passesteams)
    df_ppdabeyond40['opponents passes'] = df_ppdabeyond40['passes in game'] - df_ppdabeyond40['passes']
    df_ppdabeyond40['PPDA'] = df_ppdabeyond40['opponents passes'] / df_ppdabeyond40['defensive actions']
    df_ppda = df_ppdabeyond40[['label', 'team_name', 'PPDA']]
    return df_ppda

def create_holdsummary(df_possession_stats_summary, df_xg, df_xa,df_matchstats,df_ppda, possession_events):
    df_possession_stats_summary = pd.melt(df_possession_stats_summary, id_vars=['home_team', 'away_team', 'label'], value_vars=['home_possession', 'away_possession'], var_name='possession_type', value_name='terr_poss')
    df_possession_stats_summary['team_name'] = df_possession_stats_summary.apply(lambda x: x['home_team'] if x['possession_type'] == 'home_possession' else x['away_team'], axis=1)
    df_possession_stats_summary.drop(['home_team', 'away_team', 'possession_type'], axis=1, inplace=True)
    df_possession_stats_summary = df_possession_stats_summary[['team_name', 'label', 'terr_poss']]
    df_xg_hold = df_xg.groupby(['team_name','contestantId', 'label'])['321'].sum().reset_index()
    df_xg_hold = df_xg_hold.rename(columns={'321': 'xG'})
    df_post_shot_xg_hold = df_xg.groupby(['team_name','contestantId', 'label'])['322'].sum().reset_index()
    df_post_shot_xg_hold = df_post_shot_xg_hold.rename(columns={'322': 'Post shot xG'})
    df_cleaned_xg_hold = df_xg[df_xg['321']>0.1]
    df_cleaned_xg_hold = df_cleaned_xg_hold.groupby(['team_name','contestantId', 'label'])['321'].sum().reset_index()
    df_cleaned_xg_hold = df_cleaned_xg_hold.rename(columns={'321': 'Cleaned xG'})

    #packing_df_hold = packing_df.groupby(['teamName','label'])['bypassed_opponents'].sum().reset_index()
    #packing_df_hold = packing_df_hold.rename(columns={'teamName': 'team_name'})

    #space_control_df = space_control_df.groupby(['Team', 'label']).sum().reset_index()    
    #space_control_df['TotalControlArea_match'] = space_control_df.groupby('label')['TotalControlArea'].transform('sum')
    #space_control_df['Total Control Area %'] = space_control_df['TotalControlArea'] / space_control_df['TotalControlArea_match'] * 100
    #space_control_df = space_control_df[['Team','label','Total Control Area %']]
    #space_control_df = space_control_df.rename(columns={'Team': 'team_name'})


    df_xa_hold = df_xa.groupby(['team_name','contestantId', 'label'])['318.0'].sum().reset_index()
    df_xa_hold = df_xa_hold.rename(columns={'318.0': 'xA'})
    paentries = df_matchstats.groupby(['contestantId','label'])['penAreaEntries'].sum().reset_index()
    duels = possession_events[(possession_events['typeId'] == 44) | (possession_events['typeId'] == 7)| (possession_events['typeId'] == 45)| (possession_events['typeId'] == 3)| (possession_events['typeId'] == 4)]
    duels_team_percentage = duels.groupby(['team_name', 'label'])['outcome'].mean().reset_index(name='duel_win_percentage')
    print(duels_team_percentage)
    # Add percentage of total passes    
    duels_team_percentage = duels_team_percentage[['team_name', 'label', 'duel_win_percentage']]
    df_holdsummary = df_xa_hold.merge(df_xg_hold, how = 'outer')
    df_holdsummary = df_holdsummary.merge(paentries,how = 'outer')
    df_holdsummary = df_holdsummary.merge(df_post_shot_xg_hold,how = 'outer')
    df_holdsummary = df_holdsummary.merge(df_cleaned_xg_hold,how = 'outer')
    df_holdsummary = df_holdsummary.merge(df_possession_stats_summary,how = 'outer')
    df_holdsummary = df_holdsummary.merge(df_ppda,how = 'outer')
    df_holdsummary = df_holdsummary.merge(duels_team_percentage,how = 'outer')

    #df_holdsummary = df_holdsummary.merge(packing_df_hold)
    #df_holdsummary = df_holdsummary.merge(space_control_df)
    
    df_holdsummary = df_holdsummary[['team_name', 'label', 'xA', 'xG','Cleaned xG','Post shot xG', 'terr_poss','penAreaEntries','PPDA','duel_win_percentage']]
    return df_holdsummary

def Process_data_spillere(df_possession_xa,df_pv,df_matchstats,df_xg_all,squads):

    def calculate_score(df, column, score_column):
        df_unique = df.drop_duplicates(column).copy()
        df_unique.loc[:, score_column] = pd.qcut(df_unique[column], q=10, labels=False, duplicates='raise') + 1
        return df.merge(df_unique[[column, score_column]], on=column, how='left')

    def calculate_opposite_score(df, column, score_column):
        df_unique = df.drop_duplicates(column).copy()
        df_unique.loc[:, score_column] = pd.qcut(-df_unique[column], q=10, labels=False, duplicates='raise') + 1
        return df.merge(df_unique[[column, score_column]], on=column, how='left')

    def weighted_mean(scores, weights):
        expanded_scores = []
        for score, weight in zip(scores, weights):
            expanded_scores.extend([score] * weight)
        return np.mean(expanded_scores)

    minutter_kamp = 45
    minutter_total = 350
        
    df_possession_xa = df_possession_xa.rename(columns={'318.0': 'xA'})
    df_possession_xa_summed = df_possession_xa.groupby(['playerName','label'])['xA'].sum().reset_index()

    try:
        df_pv = df_pv_all[['playerName', 'team_name', 'label', 'possessionValue.pvValue', 'possessionValue.pvAdded']]
        df_pv.loc[:, 'possessionValue.pvValue'] = df_pv['possessionValue.pvValue'].astype(float)
        df_pv.loc[:, 'possessionValue.pvAdded'] = df_pv['possessionValue.pvAdded'].astype(float)
        df_pv['possessionValue'] = df_pv['possessionValue.pvValue'] + df_pv['possessionValue.pvAdded']
        df_kamp = df_pv.groupby(['playerName', 'label', 'team_name']).sum()
    except KeyError:
        df_pv = df_possession_xa[['playerName', 'team_name', 'label', 'xA']]
        df_pv.loc[:, 'possessionValue.pvValue'] = df_pv['xA'].astype(float)
        df_pv.loc[:, 'possessionValue.pvAdded'] = df_pv['xA'].astype(float)
        df_pv['possessionValue'] = df_pv['xA'] + df_pv['xA']
        df_kamp = df_pv.groupby(['playerName', 'label', 'team_name']).sum()

    df_kamp = df_kamp.reset_index()
    df_matchstats = df_matchstats[['player_matchName','player_playerId','contestantId','duelLost','aerialLost','player_position','player_positionSide','successfulOpenPlayPass','totalContest','duelWon','penAreaEntries','accurateBackZonePass','possWonDef3rd','wonContest','accurateFwdZonePass','openPlayPass','totalBackZonePass','minsPlayed','fwdPass','finalThirdEntries','ballRecovery','totalFwdZonePass','successfulFinalThirdPasses','totalFinalThirdPasses','attAssistOpenplay','aerialWon','totalAttAssist','possWonMid3rd','interception','totalCrossNocorner','interceptionWon','attOpenplay','touchesInOppBox','attemptsIbox','totalThroughBall','possWonAtt3rd','accurateCrossNocorner','bigChanceCreated','accurateThroughBall','totalLayoffs','accurateLayoffs','totalFastbreak','shotFastbreak','formationUsed','label','match_id','date','possLostAll','attemptsConcededIbox']]
    df_matchstats = df_matchstats.rename(columns={'player_matchName': 'playerName'})
    df_scouting = df_matchstats.merge(df_kamp)

    def calculate_match_pv(df_scouting):
        # Calculate the total match_xg for each match_id
        df_scouting['match_pv'] = df_scouting.groupby('match_id')['possessionValue.pvValue'].transform('sum')
        
        # Calculate the total team_xg for each team in each match
        df_scouting['team_pv'] = df_scouting.groupby(['contestantId', 'match_id'])['possessionValue.pvValue'].transform('sum')
        
        # Calculate opponents_xg as match_xg - team_xg
        df_scouting['opponents_pv'] = df_scouting['match_pv'] - df_scouting['team_pv']
        df_scouting['opponents_pv'] = pd.to_numeric(df_scouting['opponents_pv'], errors='coerce')
        return df_scouting
    df_scouting = calculate_match_pv(df_scouting)

    df_xg = df_xg_all[['contestantId','team_name','playerName','playerId','321','322','9','match_id','label','date']]
    df_xg = df_xg[df_xg['9']!= True]

    df_xg = df_xg.rename(columns={'321': 'xg'})
    df_xg = df_xg.rename(columns={'322': 'post shot xg'})

    df_xg['xg'] = df_xg['xg'].astype(float)
    df_xg['post shot xg'] = df_xg['post shot xg'].astype(float)

    df_xg = df_xg.groupby(['playerName','playerId','match_id','contestantId','team_name','label','date']).sum()
    df_xg = df_xg.reset_index()
    df_scouting = df_scouting.rename(columns={'player_playerId': 'playerId'})
    df_scouting = df_scouting.merge(df_xg, how='left', on=['playerName', 'playerId', 'match_id', 'contestantId', 'team_name', 'label', 'date']).reset_index()
    def calculate_match_xg(df_scouting):
        # Calculate the total match_xg for each match_id
        df_scouting['match_xg'] = df_scouting.groupby('match_id')['xg'].transform('sum')
        
        # Calculate the total team_xg for each team in each match
        df_scouting['team_xg'] = df_scouting.groupby(['contestantId', 'match_id'])['xg'].transform('sum')
        
        # Calculate opponents_xg as match_xg - team_xg
        df_scouting['opponents_xg'] = df_scouting['match_xg'] - df_scouting['team_xg']
        df_scouting['opponents_xg'] = pd.to_numeric(df_scouting['opponents_xg'], errors='coerce')
       
        return df_scouting

    df_scouting = calculate_match_xg(df_scouting)

    df_scouting = df_scouting.merge(df_possession_xa_summed, how='left')

    def calculate_match_xa(df_scouting):
        # Calculate the total match_xg for each match_id
        df_scouting['match_xA'] = df_scouting.groupby('match_id')['xA'].transform('sum')
        
        # Calculate the total team_xg for each team in each match
        df_scouting['team_xA'] = df_scouting.groupby(['contestantId', 'match_id'])['xA'].transform('sum')
        
        # Calculate opponents_xg as match_xg - team_xg
        df_scouting['opponents_xA'] = df_scouting['match_xA'] - df_scouting['team_xA']
        df_scouting['opponents_xA'] = pd.to_numeric(df_scouting['opponents_xA'], errors='coerce')
        
        return df_scouting
    
    df_scouting = calculate_match_xa(df_scouting)

    df_scouting.fillna(0, inplace=True)
    squads['dateOfBirth'] = pd.to_datetime(squads['dateOfBirth'])
    today = datetime.today()
    squads['age_today'] = ((today - squads['dateOfBirth']).dt.days / 365.25).apply(np.floor)
    squads = squads[['id','matchName','nationality','dateOfBirth','age_today']]
    squads = squads.rename(columns={'id': 'playerId'})
    squads = squads.rename(columns={'matchName': 'playerName'})
    #squads.fillna(0,inplace=True)

    df_scouting = df_scouting.merge(squads,how='outer')
    df_scouting = df_scouting.drop_duplicates(subset=['playerName', 'team_name', 'player_position', 'player_positionSide', 'label'])
    
    df_scouting['post_shot_xg_per90'] = (df_scouting['post shot xg'].astype(float) / df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['xg_per90'] = (df_scouting['xg'].astype(float) / df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['xA_per90'] = (df_scouting['xA'].astype(float) / df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['Pv_added_stoppere'] = df_scouting['possessionValue.pvValue'].astype(float).loc[df_scouting['possessionValue.pvValue'].astype(float) < 0.1]
    df_scouting['Pv_added_stoppere_per90'] = (df_scouting['Pv_added_stoppere'].astype(float) / df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['possessionValue.pvValue_per90'] = (df_scouting['possessionValue.pvValue'].astype(float) / df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['possessionValue.pvAdded_per90'] = (df_scouting['possessionValue.pvAdded'].astype(float) / df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['Possession value total per_90'] = df_scouting['possessionValue.pvAdded_per90'] + df_scouting['possessionValue.pvValue_per90']
    df_scouting['penAreaEntries_per90&crosses%shotassists'] = ((df_scouting['penAreaEntries'].astype(float)+df_scouting['totalCrossNocorner'].astype(float) + df_scouting['attAssistOpenplay'].astype(float))/ df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['penAreaEntries_per90'] = (df_scouting['penAreaEntries'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90    
    df_scouting['attAssistOpenplay_per90'] = (df_scouting['attAssistOpenplay'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['totalCrossNocorner_per90'] = (df_scouting['totalCrossNocorner'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['finalThird passes %'] = (df_scouting['successfulFinalThirdPasses'].astype(float) / df_scouting['totalFinalThirdPasses'].astype(float)) * 100
    df_scouting['finalThirdEntries_per90'] = (df_scouting['finalThirdEntries'].astype(float) / df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['interception_per90'] = (df_scouting['interception'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['possWonDef3rd_possWonMid3rd'] = (df_scouting['possWonDef3rd'].astype(float) + df_scouting['possWonMid3rd'].astype(float))
    df_scouting['possWonDef3rd_possWonMid3rd_per90'] =  (df_scouting['possWonDef3rd_possWonMid3rd'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['possWonDef3rd_possWonMid3rd_possWonAtt3rd'] = (df_scouting['possWonDef3rd'].astype(float) + df_scouting['possWonMid3rd'].astype(float) + df_scouting['possWonAtt3rd'].astype(float))
    df_scouting['possWonDef3rd_possWonMid3rd_possWonAtt3rd_per90'] =  (df_scouting['possWonDef3rd_possWonMid3rd_possWonAtt3rd'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['possWonDef3rd_possWonMid3rd_per90&interceptions_per90'] = ((df_scouting['interception_per90'].astype(float) + df_scouting['possWonDef3rd_possWonMid3rd_per90'].astype(float))/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['duels won %'] = (df_scouting['duelWon'].astype(float) / (df_scouting['duelWon'].astype(float) + df_scouting['duelLost'].astype(float)))*100
    df_scouting['Forward zone pass %'] = (df_scouting['accurateFwdZonePass'].astype(float) / df_scouting['totalFwdZonePass'].astype(float)) * 100
    df_scouting['Forward zone pass_per90'] = (df_scouting['accurateFwdZonePass'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['Back zone pass %'] = (df_scouting['accurateBackZonePass'].astype(float) / df_scouting['totalBackZonePass'].astype(float)) * 100
    df_scouting['Back zone pass_per90'] = (df_scouting['accurateBackZonePass'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['Passing %'] = (df_scouting['successfulOpenPlayPass'].astype(float) / df_scouting['openPlayPass'].astype(float)) * 100
    df_scouting['Passes_per90'] = (df_scouting['successfulOpenPlayPass'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['Duels_per90'] = (df_scouting['duelWon'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['Aerial duel %'] = (df_scouting['aerialWon'].astype(float) / (df_scouting['aerialWon'].astype(float) + df_scouting['aerialLost'].astype(float))) * 100
    df_scouting['Ballrecovery_per90'] = (df_scouting['ballRecovery'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['fwdPass_per90'] = (df_scouting['fwdPass'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['finalthirdpass_per90'] = (df_scouting['successfulFinalThirdPasses'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['shotFastbreak_per90'] = (df_scouting['shotFastbreak'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['bigChanceCreated_per90'] = (df_scouting['bigChanceCreated'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['dribble %'] = (df_scouting['wonContest'].astype(float) / df_scouting['totalContest'].astype(float)) * 100
    df_scouting['dribble_per90'] = (df_scouting['wonContest'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['touches_in_box_per90'] = (df_scouting['touchesInOppBox'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['totalThroughBall_per90'] = (df_scouting['totalThroughBall'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['attemptsIbox_per90'] = (df_scouting['attemptsIbox'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['aerialWon_per90'] = (df_scouting['aerialWon'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['possLost_per90'] = (df_scouting['possLostAll'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['Attempts conceded in box per 90'] = (df_scouting['attemptsConcededIbox'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting.fillna(0, inplace=True)

    def ball_playing_central_defender():
        df_spillende_stopper = df_scouting[(df_scouting['player_position'] == 'Defender') & (df_scouting['player_positionSide'].str.contains('Centre'))]
        df_spillende_stopper['minsPlayed'] = df_spillende_stopper['minsPlayed'].astype(int)
        df_spillende_stopper = df_spillende_stopper[df_spillende_stopper['minsPlayed'].astype(int) >= minutter_kamp]
        df_spillende_stopper = calculate_score(df_spillende_stopper,'possessionValue.pvAdded_per90', 'Possession value added score')
        df_spillende_stopper = calculate_score(df_spillende_stopper, 'duels won %', 'duels won % score')
        df_spillende_stopper = calculate_score(df_spillende_stopper, 'Forward zone pass %', 'Forward zone pass % score')
        df_spillende_stopper = calculate_score(df_spillende_stopper, 'Passing %', 'Open play passing % score')
        df_spillende_stopper = calculate_score(df_spillende_stopper, 'Back zone pass %', 'Back zone pass % score')
        df_spillende_stopper = calculate_score(df_spillende_stopper, 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score')
        df_spillende_stopper = calculate_score(df_spillende_stopper, 'Ballrecovery_per90', 'Ballrecovery_per90 score')

        df_spillende_stopper['Passing'] = df_spillende_stopper[['Open play passing % score', 'Back zone pass % score']].mean(axis=1)
        df_spillende_stopper['Forward passing'] = df_spillende_stopper[['Forward zone pass % score', 'Possession value added score', 'Possession value added score']].mean(axis=1)
        df_spillende_stopper['Defending'] = df_spillende_stopper[['duels won % score', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score', 'Ballrecovery_per90 score']].mean(axis=1)
        df_spillende_stopper['Possession value added'] = df_spillende_stopper['Possession value added score']
        
        df_spillende_stopper['Total score'] = df_spillende_stopper[['Passing','Passing','Forward passing','Forward passing','Forward passing','Defending','Defending','Possession value added','Possession value added','Possession value added']].mean(axis=1)
        df_spillende_stopper = df_spillende_stopper[['playerName','team_name','player_position','label','minsPlayed','age_today','Passing','Forward passing','Defending','Possession value added score','Total score']] 
        df_spillende_stoppertotal = df_spillende_stopper[['playerName','team_name','player_position','minsPlayed','age_today','Passing','Forward passing','Defending','Possession value added score','Total score']]
        df_spillende_stoppertotal = df_spillende_stoppertotal.groupby(['playerName','team_name','player_position','age_today']).mean().reset_index()
        minutter = df_spillende_stopper.groupby(['playerName', 'team_name','player_position','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_spillende_stoppertotal['minsPlayed total'] = minutter['minsPlayed']
        df_spillende_stopper = df_spillende_stopper.sort_values('Total score',ascending = False)
        df_spillende_stoppertotal = df_spillende_stoppertotal[['playerName','team_name','player_position','age_today','minsPlayed total','Passing','Forward passing','Defending','Possession value added score','Total score']]
        df_spillende_stoppertotal = df_spillende_stoppertotal[df_spillende_stoppertotal['minsPlayed total'].astype(int) >= minutter_total]
        df_spillende_stoppertotal = df_spillende_stoppertotal.sort_values('Total score',ascending = False)
        df_spillende_stoppertotal['Total score rank'] = df_spillende_stoppertotal['Total score'].rank(method='dense', ascending=False)
        return df_spillende_stopper
  
    def defending_central_defender():
        df_forsvarende_stopper = df_scouting[(df_scouting['player_position'] == 'Defender') & (df_scouting['player_positionSide'].str.contains('Centre'))]
        df_forsvarende_stopper['minsPlayed'] = df_forsvarende_stopper['minsPlayed'].astype(int)
        df_forsvarende_stopper = df_forsvarende_stopper[df_forsvarende_stopper['minsPlayed'].astype(int) >= minutter_kamp]
        
        df_forsvarende_stopper = calculate_score(df_forsvarende_stopper, 'duels won %', 'duels won % score')
        df_forsvarende_stopper = calculate_score(df_forsvarende_stopper, 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score')
        df_forsvarende_stopper = calculate_score(df_forsvarende_stopper, 'Ballrecovery_per90', 'ballRecovery score')
        df_forsvarende_stopper = calculate_score(df_forsvarende_stopper,'Aerial duel %', 'Aerial duel score')
        df_forsvarende_stopper = calculate_score(df_forsvarende_stopper,'possessionValue.pvAdded_per90', 'Possession value added score')
        df_forsvarende_stopper = calculate_score(df_forsvarende_stopper, 'Passing %', 'Open play passing % score')
        df_forsvarende_stopper = calculate_score(df_forsvarende_stopper, 'Back zone pass %', 'Back zone pass % score')


        df_forsvarende_stopper['Defending'] = df_forsvarende_stopper[['duels won % score','Aerial duel score', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score', 'ballRecovery score']].mean(axis=1)
        df_forsvarende_stopper['Duels'] = df_forsvarende_stopper[['duels won % score','duels won % score','Aerial duel score']].mean(axis=1)
        df_forsvarende_stopper['Intercepting'] = df_forsvarende_stopper[['possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score','possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score','ballRecovery score']].mean(axis=1)
        df_forsvarende_stopper['Passing'] = df_forsvarende_stopper[['Open play passing % score', 'Back zone pass % score','Possession value added score','Possession value added score']].mean(axis=1)
        
        df_forsvarende_stopper['Total score'] = df_forsvarende_stopper[['Defending','Defending','Defending','Defending','Duels','Duels','Duels','Intercepting','Intercepting','Intercepting','Passing','Passing']].mean(axis=1)

        df_forsvarende_stopper = df_forsvarende_stopper[['playerName','team_name','player_position','label','minsPlayed','age_today','Defending','Duels','Intercepting','Passing','Total score']]
        df_forsvarende_stoppertotal = df_forsvarende_stopper[['playerName','team_name','player_position','minsPlayed','age_today','Defending','Duels','Intercepting','Passing','Total score']]
        df_forsvarende_stoppertotal = df_forsvarende_stoppertotal.groupby(['playerName','team_name','player_position','age_today']).mean().reset_index()
        minutter = df_forsvarende_stopper.groupby(['playerName', 'team_name','player_position','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_forsvarende_stoppertotal['minsPlayed total'] = minutter['minsPlayed']
        df_forsvarende_stopper = df_forsvarende_stopper.sort_values('Total score',ascending = False)
        df_forsvarende_stoppertotal = df_forsvarende_stoppertotal[['playerName','team_name','player_position','age_today','minsPlayed total','Defending','Duels','Intercepting','Passing','Total score']]
        df_forsvarende_stoppertotal = df_forsvarende_stoppertotal[df_forsvarende_stoppertotal['minsPlayed total'].astype(int) >= minutter_total]
        df_forsvarende_stoppertotal = df_forsvarende_stoppertotal.sort_values('Total score',ascending = False)
        return df_forsvarende_stopper

    def balanced_central_defender():
        df_balanced_central_defender = df_scouting[(df_scouting['player_position'] == 'Defender') & (df_scouting['player_positionSide'].str.contains('Centre'))]
        df_balanced_central_defender['minsPlayed'] = df_balanced_central_defender['minsPlayed'].astype(int)

        # Filter players with sufficient minutes played
        df_balanced_central_defender = df_balanced_central_defender[df_balanced_central_defender['minsPlayed'] >= minutter_kamp]

        # Calculate scores
        df_balanced_central_defender = calculate_opposite_score(df_balanced_central_defender,'Attempts conceded in box per 90','Attempts conceded in box per 90 score')
        df_balanced_central_defender = calculate_opposite_score(df_balanced_central_defender, 'opponents_pv', 'opponents pv score')
        df_balanced_central_defender = calculate_opposite_score(df_balanced_central_defender, 'opponents_xg', 'opponents xg score')
        df_balanced_central_defender = calculate_opposite_score(df_balanced_central_defender, 'opponents_xA', 'opponents xA score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'duels won %', 'duels won % score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Duels_per90', 'duelWon score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Ballrecovery_per90', 'ballRecovery score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Aerial duel %', 'Aerial duel % score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'aerialWon_per90', 'Aerial duel score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Pv_added_stoppere_per90', 'Possession value added score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Passing %', 'Open play passing % score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Passes_per90', 'Passing score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Back zone pass %', 'Back zone pass % score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Back zone pass_per90', 'Back zone pass score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Forward zone pass %', 'Forward zone pass % score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Forward zone pass_per90', 'Forward zone pass score')
        df_balanced_central_defender = calculate_opposite_score(df_balanced_central_defender, 'possLost_per90', 'possLost per90 score')

        # Combine scores into categories
        df_balanced_central_defender['Defending'] = df_balanced_central_defender[
            ['duels won % score', 'duelWon score', 'opponents pv score', 'opponents xg score', 'opponents xA score',
            'opponents pv score', 'opponents xg score', 'opponents xA score', 'Aerial duel % score',
            'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score',
            'ballRecovery score','Attempts conceded in box per 90 score','Attempts conceded in box per 90 score']].mean(axis=1)
        df_balanced_central_defender['Possession value added'] = df_balanced_central_defender[
            ['Possession value added score', 'possLost per90 score']].mean(axis=1)
        df_balanced_central_defender['Passing'] = df_balanced_central_defender[
            ['Open play passing % score', 'Passing score', 'Back zone pass % score', 'Back zone pass score',
            'Back zone pass % score', 'Back zone pass score', 'Back zone pass % score', 'Back zone pass score',
            'possLost per90 score', 'possLost per90 score']].mean(axis=1)

        # Calculate component scores
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Defending', 'Defending_')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Passing', 'Passing_')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Possession value added', 'Possession_value_added')

        df_balanced_central_defender['Total score'] = df_balanced_central_defender.apply(
            lambda row: weighted_mean(
                [row['Defending_'], row['Passing_'], row['Possession_value_added']],
                [
                    7 if row['Defending_'] < 3 else 5,
                    2 if row['Passing_'] < 3 else 1,
                    1 if row['Possession_value_added'] < 3 else 1
                ]
            ),
            axis=1
        )
        
        df_balanced_central_defender = df_balanced_central_defender[
            ['playerName', 'team_name', 'player_position', 'minsPlayed','label', 'age_today', 'Defending_', 'Possession_value_added', 'Passing_', 'Total score']
        ]
        # Prepare summary
        df_balanced_central_defendertotal = df_balanced_central_defender[
            ['playerName', 'team_name', 'player_position', 'minsPlayed', 'age_today', 'Defending_', 'Possession_value_added', 'Passing_', 'Total score']
        ]
        df_balanced_central_defendertotal = df_balanced_central_defendertotal.groupby(['playerName', 'team_name', 'player_position', 'age_today']).mean().reset_index()
        df_balanced_central_defendertotal['minsPlayed total'] = df_balanced_central_defender.groupby(
            ['playerName', 'team_name', 'player_position', 'age_today'])['minsPlayed'].sum().astype(float).reset_index(drop=True)

        # Filter players with sufficient total minutes and sort
        df_balanced_central_defendertotal = df_balanced_central_defendertotal[
            df_balanced_central_defendertotal['minsPlayed total'].astype(int) >= minutter_total
        ]
        df_balanced_central_defendertotal = df_balanced_central_defendertotal.sort_values('Total score', ascending=False)
        df_balanced_central_defender = df_balanced_central_defender.sort_values('Total score', ascending=False)

        return df_balanced_central_defender
  
    def fullbacks():
        df_backs = df_scouting[((df_scouting['player_position'] == 'Defender') | (df_scouting['player_position'] == 'Wing Back')) & 
                            ((df_scouting['player_positionSide'] == 'Right') | (df_scouting['player_positionSide'] == 'Left'))]
        df_backs['minsPlayed'] = df_backs['minsPlayed'].astype(int)
        df_backs = df_backs[df_backs['minsPlayed'] >= minutter_kamp]

        df_backs = calculate_opposite_score(df_backs, 'opponents_pv', 'opponents pv score')
        df_backs = calculate_opposite_score(df_backs, 'opponents_xg', 'opponents xg score')
        df_backs = calculate_opposite_score(df_backs, 'opponents_xA', 'opponents xA score')

        df_backs = calculate_score(df_backs, 'possessionValue.pvAdded_per90', 'Possession value added score')
        df_backs = calculate_score(df_backs, 'duels won %', 'duels won % score')
        df_backs = calculate_score(df_backs, 'Duels_per90', 'Duels per 90 score')
        df_backs = calculate_score(df_backs, 'Forward zone pass %', 'Forward zone pass % score')
        df_backs = calculate_score(df_backs, 'Forward zone pass_per90', 'Forward zone pass per 90 score')
        df_backs = calculate_score(df_backs, 'penAreaEntries_per90&crosses%shotassists', 'Penalty area entries & crosses & shot assists score')
        df_backs = calculate_score(df_backs, 'attAssistOpenplay_per90', 'attAssistOpenplay_per90 score')
        df_backs = calculate_score(df_backs, 'finalThird passes %', 'finalThird passes % score')
        df_backs = calculate_score(df_backs, 'finalThirdEntries_per90', 'finalThirdEntries_per90 score')
        df_backs = calculate_score(df_backs, 'interception_per90', 'interception_per90 score')
        df_backs = calculate_score(df_backs, 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score')
        df_backs = calculate_score(df_backs, 'Back zone pass %', 'Back zone pass % score')
        df_backs = calculate_score(df_backs, 'Back zone pass_per90', 'Back zone pass_per90 score')
        df_backs = calculate_score(df_backs, 'totalCrossNocorner_per90', 'totalCrossNocorner_per90 score')
        df_backs = calculate_score(df_backs, 'xA_per90', 'xA per90 score')
        df_backs = calculate_opposite_score(df_backs, 'possLost_per90', 'possLost_per90 score')

        df_backs['Defending'] = df_backs[['opponents pv score', 'opponents xg score', 'opponents xA score', 'duels won % score',
                                        'Duels per 90 score', 'Duels per 90 score', 'duels won % score',
                                        'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score']].mean(axis=1)
        df_backs['Passing'] = df_backs[['Forward zone pass % score', 'Forward zone pass per 90 score', 'finalThird passes % score',
                                        'finalThirdEntries_per90 score', 'Back zone pass % score', 'Back zone pass_per90 score',
                                        'Possession value added score', 'possLost_per90 score', 'possLost_per90 score']].mean(axis=1)
        df_backs['Chance creation'] = df_backs[['Penalty area entries & crosses & shot assists score', 'totalCrossNocorner_per90 score',
                                                'xA per90 score', 'xA per90 score', 'finalThirdEntries_per90 score',
                                                'finalThirdEntries_per90 score', 'Forward zone pass % score',
                                                'Forward zone pass per 90 score', 'Forward zone pass per 90 score',
                                                'Forward zone pass % score', 'Possession value added score',
                                                'Possession value added score']].mean(axis=1)
        df_backs['Possession value added'] = df_backs[['Possession value added score', 'possLost_per90 score']].mean(axis=1)

        df_backs = calculate_score(df_backs, 'Defending', 'Defending_')
        df_backs = calculate_score(df_backs, 'Passing', 'Passing_')
        df_backs = calculate_score(df_backs, 'Chance creation', 'Chance_creation')
        df_backs = calculate_score(df_backs, 'Possession value added', 'Possession_value_added')

        # Calculate Total Score with Weighted Mean
        df_backs['Total score'] = df_backs.apply(
            lambda row: weighted_mean(
                [row['Defending_'], row['Passing_'], row['Chance_creation'], row['Possession_value_added']],
                [3 if row['Defending_'] < 3 else 3, 3 if row['Passing_'] < 2 else 1, 3 if row['Chance_creation'] < 3 else 2, 3 if row['Possession_value_added'] < 3 else 2]
            ), axis=1
        )

        df_backs = df_backs.dropna()

        df_backstotal = df_backs[['playerName', 'team_name', 'player_position', 'player_positionSide', 'minsPlayed',
                                'age_today', 'Defending_', 'Passing_', 'Chance_creation', 'Possession_value_added',
                                'Total score']]
        
        df_backs = df_backs[['playerName', 'team_name', 'player_position', 'player_positionSide','age_today', 'minsPlayed','label', 'Defending_', 'Passing_', 'Chance_creation','Possession_value_added', 'Total score']]

        df_backstotal = df_backstotal.groupby(['playerName', 'team_name', 'player_position', 'player_positionSide', 'age_today']).mean().reset_index()

        minutter = df_backs.groupby(['playerName', 'team_name', 'player_position', 'player_positionSide', 'age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_backstotal['minsPlayed total'] = minutter['minsPlayed']

        df_backs = df_backs.sort_values('Total score', ascending=False)
        df_backstotal = df_backstotal[['playerName', 'team_name', 'player_position', 'player_positionSide', 'age_today',
                                    'minsPlayed total', 'Defending_', 'Passing_', 'Chance_creation',
                                    'Possession_value_added', 'Total score']]
        df_backstotal = df_backstotal[df_backstotal['minsPlayed total'].astype(int) >= minutter_total]

        df_backstotal = df_backstotal.sort_values('Total score', ascending=False)

        return df_backs
    
    def number6():
        df_sekser = df_scouting[
            ((df_scouting['player_position'] == 'Defensive Midfielder') & df_scouting['player_positionSide'].str.contains('Centre')) |
            ((df_scouting['player_position'] == 'Midfielder') & df_scouting['player_positionSide'].str.contains('Centre'))]
        df_sekser.loc[:,'minsPlayed'] = df_sekser['minsPlayed'].astype(int)
        df_sekser = df_sekser[df_sekser['minsPlayed'].astype(int) >= minutter_kamp]

        df_sekser = calculate_score(df_sekser,'possessionValue.pvAdded_per90', 'Possession value added score')
        df_sekser = calculate_score(df_sekser, 'duels won %', 'duels won % score')
        df_sekser = calculate_score(df_sekser, 'Duels_per90', 'Duels per 90 score')
        df_sekser = calculate_score(df_sekser, 'Passing %', 'Passing % score')
        df_sekser = calculate_score(df_sekser, 'Passes_per90', 'Passing score')
        df_sekser = calculate_score(df_sekser, 'Back zone pass %', 'Back zone pass % score')
        df_sekser = calculate_score(df_sekser, 'Back zone pass_per90', 'Back zone pass_per90 score')
        df_sekser = calculate_score(df_sekser, 'finalThirdEntries_per90', 'finalThirdEntries_per90 score')
        df_sekser = calculate_score(df_sekser, 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score')
        df_sekser = calculate_score(df_sekser, 'possWonDef3rd_possWonMid3rd_possWonAtt3rd_per90', 'possWonDef3rd_possWonMid3rd_possWonAtt3rd_per90 score')
        df_sekser = calculate_score(df_sekser, 'Forward zone pass %', 'Forward zone pass % score')
        df_sekser = calculate_score(df_sekser, 'Forward zone pass_per90', 'Forward zone pass_per90 score')
        df_sekser = calculate_score(df_sekser, 'Ballrecovery_per90', 'ballRecovery score')
        df_sekser = calculate_opposite_score(df_sekser, 'possLost_per90', 'possLost_per90 score')

        
        df_sekser['Defending'] = df_sekser[['duels won % score','Duels per 90 score','Duels per 90 score','possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score','possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score','ballRecovery score']].mean(axis=1)
        df_sekser['Passing'] = df_sekser[['Back zone pass % score','Back zone pass_per90 score','Passing % score','Passing score','possLost_per90 score','possLost_per90 score']].mean(axis=1)
        df_sekser['Progressive ball movement'] = df_sekser[['Possession value added score','Possession value added score','Forward zone pass % score','Forward zone pass_per90 score','finalThirdEntries_per90 score']].mean(axis=1)
        df_sekser['Possession value added'] = df_sekser[['Possession value added score','possLost_per90 score']].mean(axis=1)
        
        df_sekser = calculate_score(df_sekser, 'Defending', 'Defending_')
        df_sekser = calculate_score(df_sekser, 'Passing', 'Passing_')
        df_sekser = calculate_score(df_sekser, 'Progressive ball movement','Progressive_ball_movement')
        df_sekser = calculate_score(df_sekser, 'Possession value added', 'Possession_value_added')
        
        df_sekser['Total score'] = df_sekser.apply(
        lambda row: weighted_mean(
            [row['Defending_'], row['Passing_'],row['Progressive_ball_movement'],row['Possession_value_added']],
            [3 if row['Defending_'] < 5 else 2, 3 if row['Passing_'] < 5 else 2, 3 if row['Progressive_ball_movement'] < 5 else 1, 3 if row['Possession_value_added'] < 5 else 1]
        ), axis=1
        )

        df_sekser = df_sekser[['playerName','team_name','player_position','label','minsPlayed','age_today','Defending_','Passing_','Progressive_ball_movement','Possession_value_added','Total score']]
        df_sekser = df_sekser.dropna()
        df_seksertotal = df_sekser[['playerName','team_name','player_position','minsPlayed','age_today','Defending_','Passing_','Progressive_ball_movement','Possession_value_added','Total score']]

        df_seksertotal = df_seksertotal.groupby(['playerName','team_name','player_position','age_today']).mean().reset_index()
        minutter = df_sekser.groupby(['playerName', 'team_name','player_position','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_seksertotal['minsPlayed total'] = minutter['minsPlayed']
        df_sekser = df_sekser.sort_values('Total score',ascending = False)
        df_seksertotal = df_seksertotal[['playerName','team_name','player_position','age_today','minsPlayed total','Defending_','Passing_','Progressive_ball_movement','Possession_value_added','Total score']]
        df_seksertotal= df_seksertotal[df_seksertotal['minsPlayed total'].astype(int) >= minutter_total]
        df_seksertotal = df_seksertotal.sort_values('Total score',ascending = False)

        return df_sekser

    def number6_destroyer():
        df_sekser = df_scouting[((df_scouting['player_position'] == 'Defensive Midfielder') | (df_scouting['player_position'] == 'Midfielder')) & df_scouting['player_positionSide'].str.contains('Centre')]
        df_sekser['minsPlayed'] = df_sekser['minsPlayed'].astype(int)
        df_sekser = df_sekser[df_sekser['minsPlayed'].astype(int) >= minutter_kamp]

        df_sekser = calculate_score(df_sekser,'possessionValue.pvAdded_per90', 'Possession value added score')
        df_sekser = calculate_score(df_sekser, 'duels won %', 'duels won % score')
        df_sekser = calculate_score(df_sekser, 'Passing %', 'Passing % score')
        df_sekser = calculate_score(df_sekser, 'Back zone pass %', 'Back zone pass % score')
        df_sekser = calculate_score(df_sekser, 'finalThirdEntries_per90', 'finalThirdEntries_per90 score')
        df_sekser = calculate_score(df_sekser, 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score')
        df_sekser = calculate_score(df_sekser, 'possWonDef3rd_possWonMid3rd_possWonAtt3rd_per90', 'possWonDef3rd_possWonMid3rd_possWonAtt3rd_per90 score')
        df_sekser = calculate_score(df_sekser, 'Forward zone pass %', 'Forward zone pass % score')
        df_sekser = calculate_score(df_sekser, 'Ballrecovery_per90', 'ballRecovery score')

        
        df_sekser['Defending'] = df_sekser[['duels won % score','possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score','possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score','ballRecovery score']].mean(axis=1)
        df_sekser['Passing'] = df_sekser[['Back zone pass % score','Passing % score']].mean(axis=1)
        df_sekser['Progressive ball movement'] = df_sekser[['Possession value added score','Possession value added score','Forward zone pass % score']].mean(axis=1)
        df_sekser['Possession value added'] = df_sekser['Possession value added score']
        
        df_sekser = calculate_score(df_sekser, 'Defending', 'Defending_')
        df_sekser = calculate_score(df_sekser, 'Passing', 'Passing_')
        df_sekser = calculate_score(df_sekser, 'Progressive ball movement','Progressive_ball_movement')
        df_sekser = calculate_score(df_sekser, 'Possession value added', 'Possession_value_added')
        
        df_sekser['Total score'] = df_sekser[['Defending_','Defending_','Defending_','Passing_','Passing_','Progressive_ball_movement','Possession_value_added']].mean(axis=1)
        df_sekser = df_sekser[['playerName','team_name','player_position','label','minsPlayed','age_today','Defending_','Passing_','Progressive_ball_movement','Possession_value_added','Total score']]
        df_sekser = df_sekser.dropna()

        df_seksertotal = df_sekser[['playerName','team_name','player_position','minsPlayed','age_today','Defending_','Passing_','Progressive_ball_movement','Possession_value_added','Total score']]

        df_seksertotal = df_seksertotal.groupby(['playerName','team_name','player_position','age_today']).mean().reset_index()
        minutter = df_sekser.groupby(['playerName', 'team_name','player_position','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_seksertotal['minsPlayed total'] = minutter['minsPlayed']
        df_sekser_destroyer = df_sekser.sort_values('Total score',ascending = False)
        df_seksertotal = df_seksertotal[['playerName','team_name','player_position','age_today','minsPlayed total','Defending_','Passing_','Progressive_ball_movement','Possession_value_added','Total score']]
        df_seksertotal= df_seksertotal[df_seksertotal['minsPlayed total'].astype(int) >= minutter_total]
        df_seksertotal = df_seksertotal.sort_values('Total score',ascending = False)
        return df_sekser_destroyer
    
    def number6_double_6_forward():
        df_sekser = df_scouting[((df_scouting['player_position'] == 'Defensive Midfielder') | (df_scouting['player_position'] == 'Midfielder')) & df_scouting['player_positionSide'].str.contains('Centre')]
        df_sekser['minsPlayed'] = df_sekser['minsPlayed'].astype(int)
        df_sekser = df_sekser[df_sekser['minsPlayed'].astype(int) >= minutter_kamp]

        df_sekser = calculate_score(df_sekser,'possessionValue.pvAdded_per90', 'Possession value added score')
        df_sekser = calculate_score(df_sekser, 'duels won %', 'duels won % score')
        df_sekser = calculate_score(df_sekser, 'Passing %', 'Passing % score')
        df_sekser = calculate_score(df_sekser, 'Back zone pass %', 'Back zone pass % score')
        df_sekser = calculate_score(df_sekser, 'finalThirdEntries_per90', 'finalThirdEntries_per90 score')
        df_sekser = calculate_score(df_sekser, 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score')
        df_sekser = calculate_score(df_sekser, 'possWonDef3rd_possWonMid3rd_possWonAtt3rd_per90', 'possWonDef3rd_possWonMid3rd_possWonAtt3rd_per90 score')
        df_sekser = calculate_score(df_sekser, 'Forward zone pass %', 'Forward zone pass % score')
        df_sekser = calculate_score(df_sekser, 'Ballrecovery_per90', 'ballRecovery score')

        
        df_sekser['Defending'] = df_sekser[['duels won % score','possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score','possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score','ballRecovery score']].mean(axis=1)
        df_sekser['Passing'] = df_sekser[['Back zone pass % score','Passing % score']].mean(axis=1)
        df_sekser['Progressive ball movement'] = df_sekser[['Possession value added score','Possession value added score','Forward zone pass % score']].mean(axis=1)
        df_sekser['Possession value added'] = df_sekser['Possession value added score']
        
        df_sekser = calculate_score(df_sekser, 'Defending', 'Defending_')
        df_sekser = calculate_score(df_sekser, 'Passing', 'Passing_')
        df_sekser = calculate_score(df_sekser, 'Progressive ball movement','Progressive_ball_movement')
        df_sekser = calculate_score(df_sekser, 'Possession value added', 'Possession_value_added')
        
        df_sekser['Total score'] = df_sekser[['Defending_','Defending_','Passing_','Passing_','Progressive_ball_movement','Progressive_ball_movement','Possession_value_added','Possession_value_added']].mean(axis=1)
        df_sekser = df_sekser[['playerName','team_name','player_position','label','minsPlayed','age_today','Defending_','Passing_','Progressive_ball_movement','Possession_value_added','Total score']]
        df_sekser = df_sekser.dropna()
        df_seksertotal = df_sekser[['playerName','team_name','player_position','minsPlayed','age_today','Defending_','Passing_','Progressive_ball_movement','Possession_value_added','Total score']]

        df_seksertotal = df_seksertotal.groupby(['playerName','team_name','player_position','age_today']).mean().reset_index()
        minutter = df_sekser.groupby(['playerName', 'team_name','player_position','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_seksertotal['minsPlayed total'] = minutter['minsPlayed']
        df_sekser_double_6_forward = df_sekser.sort_values('Total score',ascending = False)
        df_seksertotal = df_seksertotal[['playerName','team_name','player_position','age_today','minsPlayed total','Defending_','Passing_','Progressive_ball_movement','Possession_value_added','Total score']]
        df_seksertotal= df_seksertotal[df_seksertotal['minsPlayed total'].astype(int) >= minutter_total]
        df_seksertotal = df_seksertotal.sort_values('Total score',ascending = False)
        return df_sekser_double_6_forward
    
    def number8():
        df_otter = df_scouting[(df_scouting['player_position'] == 'Midfielder') & 
                            (df_scouting['player_positionSide'].str.contains('Centre'))]
        df_otter['minsPlayed'] = df_otter['minsPlayed'].astype(int)
        df_otter = df_otter[df_otter['minsPlayed'] >= minutter_kamp]

        # Calculate scores
        df_otter = calculate_score(df_otter, 'Possession value total per_90', 'Possession value total score')
        df_otter = calculate_score(df_otter, 'possessionValue.pvValue_per90', 'Possession value score')
        df_otter = calculate_score(df_otter, 'possessionValue.pvAdded_per90', 'Possession value added score')
        df_otter = calculate_score(df_otter, 'duels won %', 'duels won % score')
        df_otter = calculate_score(df_otter, 'Duels_per90', 'Duels per 90 score')
        df_otter = calculate_score(df_otter, 'Passing %', 'Passing % score')
        df_otter = calculate_score(df_otter, 'Passes_per90', 'Passing score')
        df_otter = calculate_score(df_otter, 'Back zone pass %', 'Back zone pass % score')
        df_otter = calculate_score(df_otter, 'Back zone pass_per90', 'Back zone pass score')
        df_otter = calculate_score(df_otter, 'finalThirdEntries_per90', 'finalThird entries score')
        df_otter = calculate_score(df_otter, 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90', 'Defensive actions score')
        df_otter = calculate_score(df_otter, 'Forward zone pass %', 'Forward zone pass % score')
        df_otter = calculate_score(df_otter, 'Forward zone pass_per90', 'Forward zone pass score')
        df_otter = calculate_score(df_otter, 'fwdPass_per90', 'Forward passes per90 score')
        df_otter = calculate_score(df_otter, 'attAssistOpenplay_per90', 'Open play assists score')
        df_otter = calculate_score(df_otter, 'penAreaEntries_per90', 'Penalty area entries score')
        df_otter = calculate_opposite_score(df_otter, 'possLost_per90', 'Possession lost per90 score')
        df_otter = calculate_score(df_otter, 'xA_per90', 'xA per90 score')

        # Combine scores into categories
        df_otter['Defending'] = df_otter[['duels won % score', 'Duels per 90 score', 'Defensive actions score']].mean(axis=1)
        df_otter['Passing'] = df_otter[['Forward zone pass % score', 'Forward zone pass score', 
                                        'Passing % score', 'Passing score', 'Possession lost per90 score']].mean(axis=1)
        df_otter['Progressive ball movement'] = df_otter[['xA per90 score', 'Forward passes per90 score', 
                                                        'Penalty area entries score', 'finalThird entries score', 
                                                        'Possession value total score', 'Possession lost per90 score']].mean(axis=1)
        df_otter['Possession value'] = df_otter[['Possession value added score', 
                                                'Possession value total score', 'Possession lost per90 score']].mean(axis=1)

        # Calculate component scores
        df_otter = calculate_score(df_otter, 'Defending', 'Defending_')
        df_otter = calculate_score(df_otter, 'Passing', 'Passing_')
        df_otter = calculate_score(df_otter, 'Progressive ball movement', 'Progressive_ball_movement')
        df_otter = calculate_score(df_otter, 'Possession value', 'Possession_value')

        # Calculate Total Score with Weighted Mean
        df_otter['Total score'] = df_otter.apply(
            lambda row: weighted_mean(
                [row['Defending_'], row['Passing_'], row['Progressive_ball_movement'], row['Possession_value']],
                [5 if row['Defending_'] > 5 else 1, 5 if row['Passing_'] > 5 else 1, 
                1 if row['Progressive_ball_movement'] < 5 else 1, 1 if row['Possession_value'] < 5 else 1]
            ), axis=1
        )

        # Prepare final output
        df_otter = df_otter.dropna()

        df_ottertotal = df_otter[['playerName', 'team_name', 'player_position', 'minsPlayed', 'age_today', 
                                'Defending_', 'Passing_', 'Progressive_ball_movement', 'Possession_value', 'Total score']]
        
        df_otter = df_otter[['playerName', 'team_name', 'player_position', 'age_today', 'minsPlayed', 'label', 
                            'Defending_', 'Passing_', 'Progressive_ball_movement', 'Possession_value', 'Total score']]

        df_ottertotal = df_ottertotal.groupby(['playerName', 'team_name', 'player_position', 'age_today']).mean().reset_index()
        minutter = df_otter.groupby(['playerName', 'team_name', 'player_position', 'age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_ottertotal['minsPlayed total'] = minutter['minsPlayed']

        df_otter = df_otter.sort_values('Total score', ascending=False)
        df_ottertotal = df_ottertotal[['playerName', 'team_name', 'player_position', 'age_today', 'minsPlayed total', 
                                    'Defending_', 'Passing_', 'Progressive_ball_movement', 'Possession_value', 'Total score']]
        df_ottertotal = df_ottertotal[df_ottertotal['minsPlayed total'].astype(int) >= minutter_total]

        df_ottertotal = df_ottertotal.sort_values('Total score', ascending=False)

        return df_otter

    def number10():
        df_10 = df_scouting[
            ((df_scouting['player_position'] == 'Attacking Midfielder') & df_scouting['player_positionSide'].str.contains('Centre'))
        ]
        df_10['minsPlayed'] = df_10['minsPlayed'].astype(int)
        df_10 = df_10[df_10['minsPlayed'] >= minutter_kamp]

        # Calculate scores
        df_10 = calculate_score(df_10, 'Possession value total per_90', 'Possession value total score')
        df_10 = calculate_score(df_10, 'possessionValue.pvValue_per90', 'Possession value score')
        df_10 = calculate_score(df_10, 'possessionValue.pvAdded_per90', 'Possession value added score')
        df_10 = calculate_score(df_10, 'Passing %', 'Passing % score')
        df_10 = calculate_score(df_10, 'Passes_per90', 'Passing score')
        df_10 = calculate_score(df_10, 'finalThirdEntries_per90', 'finalThird entries score')
        df_10 = calculate_score(df_10, 'Forward zone pass %', 'Forward zone pass % score')
        df_10 = calculate_score(df_10, 'Forward zone pass_per90', 'Forward zone pass score')
        df_10 = calculate_score(df_10, 'fwdPass_per90', 'Forward passes per90 score')
        df_10 = calculate_score(df_10, 'attAssistOpenplay_per90', 'Open play assists score')
        df_10 = calculate_score(df_10, 'penAreaEntries_per90', 'Penalty area entries score')
        df_10 = calculate_score(df_10, 'finalThird passes %', 'Final third passes % score')
        df_10 = calculate_score(df_10, 'finalthirdpass_per90', 'Final third passes per90 score')
        df_10 = calculate_score(df_10, 'dribble %', 'Dribble % score')
        df_10 = calculate_score(df_10, 'dribble_per90', 'Dribble per90 score')
        df_10 = calculate_score(df_10, 'touches_in_box_per90', 'Touches in box per90 score')
        df_10 = calculate_score(df_10, 'xA_per90', 'xA per90 score')
        df_10 = calculate_score(df_10, 'xg_per90', 'xG per90 score')
        df_10 = calculate_opposite_score(df_10, 'possLost_per90', 'Possession lost per90 score')
        df_10 = calculate_score(df_10, 'post_shot_xg_per90', 'Post shot xG per90 score')

        # Combine scores into categories
        df_10['Passing'] = df_10[['Forward zone pass % score', 'Forward zone pass score', 'Passing % score', 'Passing score']].mean(axis=1)
        df_10['Chance creation'] = df_10[['Open play assists score', 'Penalty area entries score', 'Forward zone pass % score',
                                        'Forward zone pass score', 'Final third passes % score', 'Final third passes per90 score',
                                        'Possession value total score', 'Possession value score', 'Dribble % score', 
                                        'Touches in box per90 score', 'xA per90 score']].mean(axis=1)
        df_10['Goalscoring'] = df_10[['xG per90 score', 'xG per90 score', 'xG per90 score', 'Post shot xG per90 score', 'Touches in box per90 score']].mean(axis=1)
        df_10['Possession value'] = df_10[['Possession value total score', 'Possession value added score', 'Possession value score', 
                                        'Possession lost per90 score']].mean(axis=1)

        # Calculate component scores
        df_10 = calculate_score(df_10, 'Passing', 'Passing_')
        df_10 = calculate_score(df_10, 'Chance creation', 'Chance_creation')
        df_10 = calculate_score(df_10, 'Goalscoring', 'Goalscoring_')
        df_10 = calculate_score(df_10, 'Possession value', 'Possession_value')

        # Calculate Total Score with Weighted Mean
        df_10['Total score'] = df_10.apply(
            lambda row: weighted_mean(
                [row['Passing_'], row['Chance_creation'], row['Goalscoring_'], row['Possession_value']],
                [3 if row['Passing_'] > 5 else 1, 5 if row['Chance_creation'] > 5 else 1, 
                5 if row['Goalscoring_'] > 5 else 1, 3 if row['Possession_value'] < 5 else 1]
            ), axis=1
        )

        # Prepare final output
        df_10 = df_10.dropna()

        df_10total = df_10[['playerName', 'team_name', 'minsPlayed', 'age_today', 
                            'Passing_', 'Chance_creation', 'Goalscoring_', 'Possession_value', 'Total score']]
        
        df_10 = df_10[['playerName', 'team_name', 'age_today', 'minsPlayed', 'label', 
                    'Passing_', 'Chance_creation', 'Goalscoring_', 'Possession_value', 'Total score']]

        df_10total = df_10total.groupby(['playerName', 'team_name', 'age_today']).mean().reset_index()
        minutter = df_10.groupby(['playerName', 'team_name', 'age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_10total['minsPlayed total'] = minutter['minsPlayed']

        df_10 = df_10.sort_values('Total score', ascending=False)
        df_10total = df_10total[['playerName', 'team_name', 'age_today', 'minsPlayed total', 
                                'Passing_', 'Chance_creation', 'Goalscoring_', 'Possession_value', 'Total score']]
        df_10total = df_10total[df_10total['minsPlayed total'].astype(int) >= minutter_total]

        df_10total = df_10total.sort_values('Total score', ascending=False)

        return df_10

    def winger():
        df_winger = df_scouting[
            (
                (df_scouting['player_position'] == 'Midfielder') &
                (df_scouting['player_positionSide'].isin(['Right', 'Left']))
            ) |
            (
                (df_scouting['player_position'].isin(['Attacking Midfielder', 'Striker'])) &
                (df_scouting['player_positionSide'].isin(['Right', 'Left']))
            )
        ]
        df_winger['minsPlayed'] = df_winger['minsPlayed'].astype(int)
        df_winger = df_winger[df_winger['minsPlayed'] >= minutter_kamp]

        # Calculate scores
        df_winger = calculate_score(df_winger, 'Possession value total per_90', 'Possession value total score')
        df_winger = calculate_score(df_winger, 'possessionValue.pvValue_per90', 'Possession value score')
        df_winger = calculate_score(df_winger, 'possessionValue.pvAdded_per90', 'Possession value added score')
        df_winger = calculate_score(df_winger, 'Passing %', 'Passing % score')
        df_winger = calculate_score(df_winger, 'Passes_per90', 'Passing score')
        df_winger = calculate_score(df_winger, 'finalThirdEntries_per90', 'Final third entries score')
        df_winger = calculate_score(df_winger, 'Forward zone pass %', 'Forward zone pass % score')
        df_winger = calculate_score(df_winger, 'Forward zone pass_per90', 'Forward zone pass score')
        df_winger = calculate_score(df_winger, 'fwdPass_per90', 'Forward passes per90 score')
        df_winger = calculate_score(df_winger, 'attAssistOpenplay_per90', 'Open play assists score')
        df_winger = calculate_score(df_winger, 'penAreaEntries_per90', 'Penalty area entries score')
        df_winger = calculate_score(df_winger, 'finalThird passes %', 'Final third passes % score')
        df_winger = calculate_score(df_winger, 'finalthirdpass_per90', 'Final third passes per90 score')
        df_winger = calculate_score(df_winger, 'dribble %', 'Dribble % score')
        df_winger = calculate_score(df_winger, 'dribble_per90', 'Dribble per90 score')
        df_winger = calculate_score(df_winger, 'touches_in_box_per90', 'Touches in box per90 score')
        df_winger = calculate_score(df_winger, 'xA_per90', 'xA per90 score')
        df_winger = calculate_score(df_winger, 'attemptsIbox_per90', 'Attempts in box per90 score')
        df_winger = calculate_score(df_winger, 'xg_per90', 'xG per90 score')
        df_winger = calculate_score(df_winger, 'post_shot_xg_per90', 'Post shot xG per90 score')

        # Combine scores into categories
        df_winger['Passing'] = df_winger[['Forward zone pass % score', 'Forward zone pass score', 'Passing % score', 'Passing score']].mean(axis=1)
        df_winger['Chance creation'] = df_winger[['Open play assists score', 'Penalty area entries score', 'Forward zone pass % score', 
                                                'Forward zone pass score', 'Final third passes % score', 'Final third passes per90 score', 
                                                'Possession value total score', 'Possession value score', 'Dribble % score', 
                                                'Dribble per90 score', 'Touches in box per90 score', 'xA per90 score']].mean(axis=1)
        df_winger['Goalscoring'] = df_winger[['xG per90 score', 'xG per90 score', 'xG per90 score', 'Post shot xG per90 score', 'Touches in box per90 score']].mean(axis=1)
        df_winger['Possession value'] = df_winger[['Possession value total score', 'Possession value added score', 'Possession value score']].mean(axis=1)

        # Calculate component scores
        df_winger = calculate_score(df_winger, 'Passing', 'Passing_')
        df_winger = calculate_score(df_winger, 'Chance creation', 'Chance_creation')
        df_winger = calculate_score(df_winger, 'Goalscoring', 'Goalscoring_')
        df_winger = calculate_score(df_winger, 'Possession value', 'Possession_value')

        # Calculate Total Score with Weighted Mean
        df_winger['Total score'] = df_winger.apply(
            lambda row: weighted_mean(
                [row['Passing_'], row['Chance_creation'], row['Goalscoring_'], row['Possession_value']],
                [1 if row['Passing_'] > 5 else 1, 5 if row['Chance_creation'] > 5 else 1, 
                5 if row['Goalscoring_'] > 5 else 1, 3 if row['Possession_value'] > 5 else 1]
            ), axis=1
        )

        # Prepare final output
        df_winger = df_winger.dropna()
        
        df_winger = df_winger[['playerName', 'team_name', 'age_today', 'minsPlayed', 'label', 
                    'Passing_', 'Chance_creation', 'Goalscoring_', 'Possession_value', 'Total score']]

        df_winger_total = df_winger[['playerName', 'team_name', 'minsPlayed', 
                                    'age_today', 'Passing_', 'Chance_creation', 'Goalscoring_', 'Possession_value', 'Total score']]
        df_winger_total = df_winger_total.groupby(['playerName', 'team_name', 'age_today']).mean().reset_index()
        minutter = df_winger.groupby(['playerName', 'team_name', 'age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_winger_total['minsPlayed total'] = minutter['minsPlayed']

        df_winger_total = df_winger_total[df_winger_total['minsPlayed total'].astype(int) >= minutter_total]
        df_winger_total = df_winger_total.sort_values('Total score', ascending=False)
        df_winger = df_winger.sort_values('Total score', ascending=False)

        return df_winger

    def Classic_striker():
        df_striker = df_scouting[(df_scouting['player_position'] == 'Striker') & (df_scouting['player_positionSide'].str.contains('Centre'))]
        df_striker['minsPlayed'] = df_striker['minsPlayed'].astype(int)
        df_striker = df_striker[df_striker['minsPlayed'].astype(int) >= minutter_kamp]

        df_striker = calculate_score(df_striker,'Possession value total per_90','Possession value total score')
        df_striker = calculate_score(df_striker,'possessionValue.pvValue_per90', 'Possession value score')
        df_striker = calculate_score(df_striker,'possessionValue.pvAdded_per90', 'Possession value added score')
        df_striker = calculate_score(df_striker, 'Passing %', 'Passing % score')
        df_striker = calculate_score(df_striker, 'Passes_per90', 'Passing score')
        df_striker = calculate_score(df_striker, 'finalThirdEntries_per90', 'finalThirdEntries_per90 score')
        df_striker = calculate_score(df_striker, 'Forward zone pass %', 'Forward zone pass % score')
        df_striker = calculate_score(df_striker, 'Forward zone pass_per90', 'Forward zone pass score')
        df_striker = calculate_score(df_striker, 'fwdPass_per90', 'fwd_Pass_per90 score')
        df_striker = calculate_score(df_striker, 'attAssistOpenplay_per90','attAssistOpenplay_per90 score')
        df_striker = calculate_score(df_striker, 'penAreaEntries_per90','penAreaEntries_per90 score')
        df_striker = calculate_score(df_striker, 'finalThird passes %','finalThird passes % score')
        df_striker = calculate_score(df_striker, 'finalthirdpass_per90','finalThird passes per90 score')
        df_striker = calculate_score(df_striker, 'shotFastbreak_per90','shotFastbreak_per90 score')
        df_striker = calculate_score(df_striker, 'dribble %','dribble % score')
        df_striker = calculate_score(df_striker, 'dribble_per90', 'dribble_per90 score')
        df_striker = calculate_score(df_striker, 'touches_in_box_per90','touches_in_box_per90 score')
        df_striker = calculate_score(df_striker, 'xA_per90','xA_per90 score')
        df_striker = calculate_score(df_striker, 'attemptsIbox_per90','attemptsIbox_per90 score')
        df_striker = calculate_score(df_striker, 'xg_per90','xg_per90 score')
        df_striker = calculate_score(df_striker, 'post_shot_xg_per90','post_shot_xg_per90 score')


        df_striker['Linkup_play'] = df_striker[['Forward zone pass % score','Forward zone pass score','Passing % score','Passing score','Possession value score','penAreaEntries_per90 score','finalThirdEntries_per90 score']].mean(axis=1)
        df_striker['Chance_creation'] = df_striker[['penAreaEntries_per90 score','Possession value total score','touches_in_box_per90 score','finalThirdEntries_per90 score']].mean(axis=1)
        df_striker['Goalscoring_'] = df_striker[['post_shot_xg_per90','xg_per90 score','xg_per90 score','xg_per90 score']].mean(axis=1)
        df_striker['Possession_value'] = df_striker[['Possession value total score','Possession value score','Possession value score','Possession value score']].mean(axis=1)

        df_striker = calculate_score(df_striker, 'Linkup_play', 'Linkup play')
        df_striker = calculate_score(df_striker, 'Chance_creation','Chance creation')
        df_striker = calculate_score(df_striker, 'Goalscoring_','Goalscoring')        
        df_striker = calculate_score(df_striker, 'Possession_value', 'Possession value')

        df_striker['Total score'] = df_striker.apply(
            lambda row: weighted_mean(
                [row['Linkup play'], row['Chance creation'], row['Goalscoring'], row['Possession value']],
                [3 if row['Linkup play'] > 5 else 1, 3 if row['Chance creation'] > 5 else 1, 
                5 if row['Goalscoring'] > 5 else 2, 3 if row['Possession value'] < 5 else 1]
            ), axis=1
        )        
        df_striker = df_striker.dropna()
        df_striker= df_striker[['playerName', 'team_name', 'age_today', 'minsPlayed', 'label', 
                    'Linkup play', 'Chance creation', 'Goalscoring', 'Possession value', 'Total score']]

        df_striker_total = df_striker[['playerName', 'team_name', 'minsPlayed', 
                                    'age_today', 'Linkup play', 'Chance creation', 'Goalscoring', 'Possession value', 'Total score']]
        df_striker_total = df_striker_total.groupby(['playerName', 'team_name', 'age_today']).mean().reset_index()
        minutter = df_striker.groupby(['playerName', 'team_name', 'age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_striker_total['minsPlayed total'] = minutter['minsPlayed']

        df_striker_total = df_striker_total[df_striker_total['minsPlayed total'].astype(int) >= minutter_total]
        df_striker_total = df_striker_total.sort_values('Total score', ascending=False)
        df_striker = df_striker.sort_values('Total score',ascending = False)
        return df_striker

    def Targetman():
        df_striker = df_scouting[(df_scouting['player_position'] == 'Striker') & (df_scouting['player_positionSide'].str.contains('Centre'))]
        df_striker['minsPlayed'] = df_striker['minsPlayed'].astype(int)
        df_striker = df_striker[df_striker['minsPlayed'].astype(int) >= minutter_kamp]

        df_striker = calculate_score(df_striker,'Possession value total per_90','Possession value total score')
        df_striker = calculate_score(df_striker,'possessionValue.pvValue_per90', 'Possession value score')
        df_striker = calculate_score(df_striker,'possessionValue.pvAdded_per90', 'Possession value added score')
        df_striker = calculate_score(df_striker, 'Passing %', 'Passing % score')
        df_striker = calculate_score(df_striker, 'finalThirdEntries_per90', 'finalThirdEntries_per90 score')
        df_striker = calculate_score(df_striker, 'Forward zone pass %', 'Forward zone pass % score')
        df_striker = calculate_score(df_striker, 'fwdPass_per90', 'fwd_Pass_per90 score')
        df_striker = calculate_score(df_striker, 'attAssistOpenplay_per90','attAssistOpenplay_per90 score')
        df_striker = calculate_score(df_striker, 'penAreaEntries_per90','penAreaEntries_per90 score')
        df_striker = calculate_score(df_striker, 'finalThird passes %','finalThird passes % score')
        df_striker = calculate_score(df_striker, 'shotFastbreak_per90','shotFastbreak_per90 score')
        df_striker = calculate_score(df_striker, 'bigChanceCreated_per90','bigChanceCreated_per90 score')
        df_striker = calculate_score(df_striker, 'dribble %','dribble % score')
        df_striker = calculate_score(df_striker, 'touches_in_box_per90','touches_in_box_per90 score')
        df_striker = calculate_score(df_striker, 'xA_per90','xA_per90 score')
        df_striker = calculate_score(df_striker, 'attemptsIbox_per90','attemptsIbox_per90 score')
        df_striker = calculate_score(df_striker, 'xg_per90','xg_per90 score')
        df_striker = calculate_score(df_striker, 'aerialWon','aerialWon score')


        df_striker['Linkup_play'] = df_striker[['Forward zone pass % score','Passing % score','Possession value score','penAreaEntries_per90 score','finalThirdEntries_per90 score','aerialWon score']].mean(axis=1)
        df_striker['Chance_creation'] = df_striker[['penAreaEntries_per90 score','Possession value total score','bigChanceCreated_per90 score','touches_in_box_per90 score','finalThirdEntries_per90 score']].mean(axis=1)
        df_striker['Goalscoring_'] = df_striker[['attemptsIbox_per90 score','xg_per90 score','xg_per90 score','xg_per90 score','xg_per90 score']].mean(axis=1)
        df_striker['Possession_value'] = df_striker[['Possession value total score','Possession value score','Possession value score','Possession value score']].mean(axis=1)

        df_striker = calculate_score(df_striker, 'Linkup_play', 'Linkup play')
        df_striker = calculate_score(df_striker, 'Chance_creation','Chance creation')
        df_striker = calculate_score(df_striker, 'Goalscoring_','Goalscoring')        
        df_striker = calculate_score(df_striker, 'Possession_value', 'Possession value')

        
        df_striker['Total score'] = df_striker[['Linkup play','Linkup play','Linkup play','Chance creation','Goalscoring','Goalscoring','Possession value','Possession value']].mean(axis=1)
        df_striker = df_striker[['playerName','team_name','label','minsPlayed','age_today','Linkup play','Chance creation','Goalscoring','Possession value','Total score']]
        df_striker = df_striker.dropna()
        df_strikertotal = df_striker[['playerName','team_name','minsPlayed','age_today','Linkup play','Chance creation','Goalscoring','Possession value','Total score']]

        df_strikertotal = df_strikertotal.groupby(['playerName','team_name','age_today']).mean().reset_index()
        minutter = df_striker.groupby(['playerName', 'team_name','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_strikertotal['minsPlayed total'] = minutter['minsPlayed']
        df_targetman = df_striker.sort_values('Total score',ascending = False)
        df_strikertotal = df_strikertotal[['playerName','team_name','age_today','minsPlayed total','Linkup play','Chance creation','Goalscoring','Possession value','Total score']]
        df_strikertotal= df_strikertotal[df_strikertotal['minsPlayed total'].astype(int) >= minutter_total]
        df_strikertotal = df_strikertotal.sort_values('Total score',ascending = False)
        return df_targetman

    def Boxstriker():
        df_striker = df_scouting[(df_scouting['player_position'] == 'Striker') & (df_scouting['player_positionSide'].str.contains('Centre'))]
        df_striker['minsPlayed'] = df_striker['minsPlayed'].astype(int)
        df_striker = df_striker[df_striker['minsPlayed'].astype(int) >= minutter_kamp]

        df_striker = calculate_score(df_striker,'Possession value total per_90','Possession value total score')
        df_striker = calculate_score(df_striker,'possessionValue.pvValue_per90', 'Possession value score')
        df_striker = calculate_score(df_striker,'possessionValue.pvAdded_per90', 'Possession value added score')
        df_striker = calculate_score(df_striker, 'Passing %', 'Passing % score')
        df_striker = calculate_score(df_striker, 'finalThirdEntries_per90', 'finalThirdEntries_per90 score')
        df_striker = calculate_score(df_striker, 'Forward zone pass %', 'Forward zone pass % score')
        df_striker = calculate_score(df_striker, 'fwdPass_per90', 'fwd_Pass_per90 score')
        df_striker = calculate_score(df_striker, 'attAssistOpenplay_per90','attAssistOpenplay_per90 score')
        df_striker = calculate_score(df_striker, 'penAreaEntries_per90','penAreaEntries_per90 score')
        df_striker = calculate_score(df_striker, 'finalThird passes %','finalThird passes % score')
        df_striker = calculate_score(df_striker, 'shotFastbreak_per90','shotFastbreak_per90 score')
        df_striker = calculate_score(df_striker, 'bigChanceCreated_per90','bigChanceCreated_per90 score')
        df_striker = calculate_score(df_striker, 'dribble %','dribble % score')
        df_striker = calculate_score(df_striker, 'touches_in_box_per90','touches_in_box_per90 score')
        df_striker = calculate_score(df_striker, 'xA_per90','xA_per90 score')
        df_striker = calculate_score(df_striker, 'attemptsIbox_per90','attemptsIbox_per90 score')
        df_striker = calculate_score(df_striker, 'xg_per90','xg_per90 score')


        df_striker['Linkup_play'] = df_striker[['Forward zone pass % score','Passing % score','Possession value score','penAreaEntries_per90 score','finalThirdEntries_per90 score']].mean(axis=1)
        df_striker['Chance_creation'] = df_striker[['penAreaEntries_per90 score','Possession value total score','bigChanceCreated_per90 score','touches_in_box_per90 score','finalThirdEntries_per90 score']].mean(axis=1)
        df_striker['Goalscoring_'] = df_striker[['attemptsIbox_per90 score','xg_per90 score','xg_per90 score','xg_per90 score','xg_per90 score']].mean(axis=1)
        df_striker['Possession_value'] = df_striker[['Possession value total score','Possession value score','Possession value score','Possession value score']].mean(axis=1)

        df_striker = calculate_score(df_striker, 'Linkup_play', 'Linkup play')
        df_striker = calculate_score(df_striker, 'Chance_creation','Chance creation')
        df_striker = calculate_score(df_striker, 'Goalscoring_','Goalscoring')        
        df_striker = calculate_score(df_striker, 'Possession_value', 'Possession value')

        
        df_striker['Total score'] = df_striker[['Linkup play','Chance creation','Goalscoring','Goalscoring','Goalscoring','Goalscoring','Possession value','Possession value','Possession value']].mean(axis=1)
        df_striker = df_striker[['playerName','team_name','label','minsPlayed','age_today','Linkup play','Chance creation','Goalscoring','Possession value','Total score']]
        df_striker = df_striker.dropna()
        df_strikertotal = df_striker[['playerName','team_name','minsPlayed','age_today','Linkup play','Chance creation','Goalscoring','Possession value','Total score']]

        df_strikertotal = df_strikertotal.groupby(['playerName','team_name','age_today']).mean().reset_index()
        minutter = df_striker.groupby(['playerName', 'team_name','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_strikertotal['minsPlayed total'] = minutter['minsPlayed']
        df_boksstriker = df_striker.sort_values('Total score',ascending = False)
        df_strikertotal = df_strikertotal[['playerName','team_name','age_today','minsPlayed total','Linkup play','Chance creation','Goalscoring','Possession value','Total score']]
        df_strikertotal= df_strikertotal[df_strikertotal['minsPlayed total'].astype(int) >= minutter_total]
        df_strikertotal = df_strikertotal.sort_values('Total score',ascending = False)
        return df_boksstriker

    return {
        'Central defender': balanced_central_defender(),
        'Fullbacks': fullbacks(),
        'Number 6' : number6(),
        'Number 8': number8(),
        'Number 10': number10(),
        'Winger': winger(),
        'Classic striker': Classic_striker(),
    }

def process_data():
    expected_points_xg, total_expected_points_xg = calculate_expected_points(df_xg, '321')
    expected_points_xg_top_6, total_expected_points_xg_top_6 = calculate_expected_points_top_6(df_xg, '321')

    # Calculate expected points based on xA
    expected_points_xa, total_expected_points_xa = calculate_expected_points(df_xa, '318.0')
    expected_points_xa_top_6, total_expected_points_xa_top_6 = calculate_expected_points_top_6(df_xa, '318.0')

    df_holdsummary = create_holdsummary(df_possession_stats_summary, df_xg, df_xa, df_matchstats,df_ppda, possession_events)
    # Merge the expected points from both xG and xA simulations
    merged_df = expected_points_xg.merge(expected_points_xa, on=['label','date', 'team_name'], suffixes=('_xg', '_xa'))
    merged_df['expected_points'] = (merged_df['expected_points_xg'] + merged_df['expected_points_xa']) / 2
    merged_df['win_probability'] = (merged_df['win_probability_xg'] + merged_df['win_probability_xa']) / 2
    merged_df['draw_probability'] = (merged_df['draw_probability_xg'] + merged_df['draw_probability_xa']) / 2
    merged_df['loss_probability'] = (merged_df['loss_probability_xg'] + merged_df['loss_probability_xa']) / 2
    merged_df = merged_df.merge(df_holdsummary,on=['label', 'team_name'],how='outer')
    label_counts_per_team = merged_df.groupby('team_name')['label'].count().reset_index()
    top_6_teams = ['Hvidovre', 'Horsens', 'Fredericia', 'Esbjerg', 'OB', 'Kolding']

    # Find labels hvor mindst to af holdene er blandt top 6
    labels_with_two_or_more_top_6 = (
        merged_df[merged_df['team_name'].isin(top_6_teams)]
        .groupby('label')['team_name']
        .nunique()
        .reset_index()
    )
    labels_with_two_or_more_top_6 = labels_with_two_or_more_top_6[labels_with_two_or_more_top_6['team_name'] >= 2]['label']

    # Filtrer original dataframe til kun de labels
    filtered_df = merged_df[merged_df['label'].isin(labels_with_two_or_more_top_6)]

    # Tæl kampe per hold (kun de labels med to eller flere af de nævnte hold)
    label_counts_per_team_top_6 = (
        filtered_df.groupby('team_name')['label']
        .count()
        .reset_index()
        .rename(columns={'label': 'Top 6 match count'})
    )
    horsens_df = merged_df[merged_df['team_name'] == 'Horsens']

    total_expected_points_combined = total_expected_points_xg.merge(total_expected_points_xa, on='team_name', suffixes=('_xg', '_xa'))
    total_expected_points_combined['Total expected points'] = (total_expected_points_combined['total_expected_points_xg'] + total_expected_points_combined['total_expected_points_xa']) / 2
    total_expected_points_combined = total_expected_points_combined[['team_name', 'Total expected points']]
    total_expected_points_combined = label_counts_per_team.merge(total_expected_points_combined)
    total_expected_points_combined ['Expected points per game'] = total_expected_points_combined['Total expected points'] / total_expected_points_combined['label']
    total_expected_points_combined = total_expected_points_combined.rename(columns={'label': 'matches'})

    total_expected_points_combined_top_6 = total_expected_points_xg_top_6.merge(total_expected_points_xa_top_6, on='team_name', suffixes=('_xg', '_xa'))
    total_expected_points_combined_top_6['Total expected points'] = (total_expected_points_combined_top_6['total_expected_points_xg'] + total_expected_points_combined_top_6['total_expected_points_xa']) / 2
    total_expected_points_combined_top_6 = total_expected_points_combined_top_6[['team_name', 'Total expected points']]
    total_expected_points_combined_top_6 = total_expected_points_combined_top_6[total_expected_points_combined_top_6['Total expected points'] > 0]
    total_expected_points_combined_top_6 = label_counts_per_team_top_6.merge(total_expected_points_combined_top_6)
    total_expected_points_combined_top_6 ['Expected points per game'] = total_expected_points_combined_top_6['Total expected points'] / total_expected_points_combined_top_6['Top 6 match count']

    return horsens_df, merged_df, total_expected_points_combined,total_expected_points_combined_top_6


df_xg, df_xa, df_pv, df_possession_stats, df_xa_agg, df_possession_data, df_xg_agg, df_pv_agg, df_xg_all, df_possession_xa, df_pv_all, df_matchstats, squads, possession_events = load_data()
df_xg_agg, df_xa_agg, df_pv_agg, df_possession_stats, df_possession_stats_summary = preprocess_data(df_xg_agg, df_xa_agg, df_pv_agg, df_possession_stats)
df_ppda = calculate_ppda(df_possession_data)

horsens_df, merged_df, total_expected_points_combined,total_expected_points_combined_top_6 = process_data()

position_dataframes = Process_data_spillere(df_possession_xa, df_pv, df_matchstats, df_xg_all, squads)

#defending_central_defender_df = position_dataframes['defending_central_defender']
#ball_playing_central_defender_df = position_dataframes['ball_playing_central_defender']
balanced_central_defender_df = position_dataframes['Central defender']
fullbacks_df = position_dataframes['Fullbacks']
number6_df = position_dataframes['Number 6']
#number6_double_6_forward_df = position_dataframes['number6_double_6_forward']
#number6_destroyer_df = position_dataframes['Number 6 (destroyer)']
number8_df = position_dataframes['Number 8']
number10_df = position_dataframes['Number 10']
winger_df = position_dataframes['Winger']
classic_striker_df = position_dataframes['Classic striker']
#targetman_df = position_dataframes['Targetman']
#box_striker_df = position_dataframes['Boxstriker']
horsens_df, merged_df, total_expected_points_combined,total_expected_points_combined_top_6 = process_data()


def create_pdf_game_report(game_data, df_xg_agg, df_xa_agg, merged_df, df_possession_stats, position_dataframes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    label = game_data['label']
    expected_points = game_data['expected_points']
    win_prob = game_data['win_probability']
    draw_prob = game_data['draw_probability']
    loss_prob = game_data['loss_probability']
    
    # Add the team logo
    pdf.image('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/Logo.png', x=165, y=5, w=15, h=15)
    pdf.set_xy(10, 10)
    pdf.cell(140, 5, txt=f"Match Report: {label}", ln=True, align='L')
    
    pdf.ln(4)
    
    # Create bar charts for expected points and probabilities
    create_bar_chart(expected_points, 'Expected Points', 'bar_combined.png', 3.0, [1.0, 1.3, 1.8], ['Relegation','Top 6', 'Promotion'])
    create_stacked_bar_chart(win_prob, draw_prob, loss_prob, 'Win/Draw/Loss Probabilities', 'bar_combined_win_prob.png')
    
    # Add bar charts to PDF side by side
    pdf.image('bar_combined.png', x=10, y=25, w=90, h=30)
    pdf.image('bar_combined_win_prob.png', x=110, y=25, w=90, h=30)

    pdf.set_xy(5, 60)

    game_xg_agg = df_xg_agg[df_xg_agg['label'] == label]
    game_xa_agg = df_xa_agg[df_xa_agg['label'] == label]
    df_possession_stats = df_possession_stats[df_possession_stats['label'] == label]
    first_home_team = df_possession_stats['home_team'].iloc[0]
    first_away_team = df_possession_stats['away_team'].iloc[0]
    df_possession_stats = df_possession_stats.rename(columns={'home': first_home_team, 'away': first_away_team})
    generate_cumulative_chart(game_xg_agg, 'cumulativxg', 'xG', 'cumulative_xg.png')
    generate_cumulative_chart(game_xa_agg, 'cumulativxa', 'xA', 'cumulative_xa.png')
    generate_possession_chart(df_possession_stats, 'cumulative_possession.png')

    pdf.image('cumulative_xg.png', x=5, y=55, w=66, h=60)
    pdf.image('cumulative_xa.png', x=72, y=55, w=66, h=60)
    pdf.image('cumulative_possession.png', x=139, y=55, w=66, h=60)

    # Add a summary table
    pdf.set_xy(5, 115)
    pdf.set_font_size(6)
    pdf.cell(20, 5, 'Summary', 0, 1, 'C')
    pdf.set_font("Arial", size=6)
    pdf.cell(20, 5, 'Team', 1)
    pdf.cell(10, 5, 'xA', 1)
    pdf.cell(10, 5, 'xG', 1)
    pdf.cell(20, 5, 'Cleaned xG', 1)
    pdf.cell(20, 5, 'Post shot xG', 1)
    pdf.cell(30, 5, 'Territorial possession', 1)
    pdf.cell(25, 5, 'Penalty area entries', 1)
    pdf.cell(10, 5, 'PPDA', 1)
    pdf.cell(20, 5, 'Duels (%)', 1)

    pdf.ln()

    game_merged_df = merged_df[merged_df['label'] == label]
    for index, row in game_merged_df.iterrows():
        pdf.cell(20, 5, row['team_name'], 1)
        pdf.cell(10, 5, f"{row['xA']:.2f}", 1)
        pdf.cell(10, 5, f"{row['xG']:.2f}", 1)
        pdf.cell(20, 5, f"{row['Cleaned xG']:.2f}", 1)
        pdf.cell(20, 5, f"{row['Post shot xG']:.2f}", 1)
        pdf.cell(30, 5, f"{row['terr_poss']:.2f}", 1)
        pdf.cell(25, 5, f"{row['penAreaEntries']:.0f}", 1)
        pdf.cell(10, 5, f"{row['PPDA']:.2f}", 1)
        pdf.cell(15, 5, f"{row['duel_win_percentage']:.2f}", 1)

        pdf.ln()
        
    for position, df in position_dataframes.items():
        filtered_df = df[(df['team_name'] == 'Horsens') & (df['label'] == label)]
        if 'player_position' in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=['label', 'team_name', 'player_position', 'age_today'])
        else:
            filtered_df = filtered_df.drop(columns=['label', 'team_name', 'age_today'])

        if 'player_positionSide' in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=['player_positionSide'])
        else:
            filtered_df = filtered_df
        filtered_df = filtered_df.round(2)
        filtered_df['Total score'] = filtered_df['Total score'].astype(float)
        pdf.set_font("Arial", size=6)
        pdf.cell(190, 4, txt=f"Position Report: {position}", ln=True, align='C')
        pdf.ln(1)

        # Add table headers
        pdf.set_font("Arial", size=6)
        headers = filtered_df.columns
        col_width = 30  # Fixed width for all columns except the last one
        last_col_width = 15  # Width for the last column

        for header in headers[:-1]:
            pdf.cell(col_width, 4, txt=header, border=1)
        pdf.cell(last_col_width, 4, txt=headers[-1], border=1)
        pdf.ln(4)

        # Add table content
        for index, row in filtered_df.iterrows():
            total_score = row['Total score']
            if total_score < 4:
                fill_color = (255, 0, 0)  # Red
            elif 4 <= total_score <= 6:
                fill_color = (255, 255, 0)  # Yellow
            else:
                fill_color = (0, 255, 0)  # Green

            pdf.set_fill_color(*fill_color)

            # Add all cells for the row
            for value in row.values[:-1]:
                pdf.cell(col_width, 4, txt=str(value), border=1, fill=True)
            pdf.cell(last_col_width, 4,txt=str(row.values[-1]), border=1, fill=True)
            pdf.ln(4)       
    pdf.output(f"Match reports/Match_Report_{label}.pdf")
    print(f'{label} report created')
# Generate a PDF report for each game involving Horsens
for index, row in horsens_df.iterrows():
    # Define the file path based on the label
    label = row['label']  # Adjust this to the correct column name for your label
    file_path = f"Match reports/Match_Report_{label}.pdf"
    
    # Check if the file already exists
    if not os.path.exists(file_path):
    # If it doesn't exist, create the PDF
        create_pdf_game_report(row, df_xg_agg, df_xa_agg, merged_df, df_possession_stats, position_dataframes)
    else:
        print(f"Skipping creation for {label}, PDF already exists.")

def create_pdf_progress_report(horsens_df, total_expected_points_combined,total_expected_points_combined_top_6, position_dataframes):
    today = date.today()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Add the team logo
    pdf.image('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/Logo.png', x=165, y=5, w=10, h=10)
    pdf.set_xy(5, 5)
    pdf.cell(140, 5, txt=f"Progress report: {today}", ln=True, align='L')

    # Save the sliding average plot as an image
    plt.figure(figsize=(12, 3))
    sliding_average_plot(horsens_df, window_size=3, filename='sliding_average_plot.png')
    plt.close()
    pdf.image("sliding_average_plot.png", x=5, y=15, w=200)

    # Generate the DataFrame summary table for total expected points
    total_expected_points_combined = total_expected_points_combined.round(2)
    total_expected_points_combined = total_expected_points_combined.sort_values(by='Expected points per game', ascending=False)
    total_expected_points_combined = total_expected_points_combined[total_expected_points_combined['team_name'].isin(['Kolding','Fredericia','Horsens','OB','Hvidovre','Esbjerg'])]
    plt.figure(figsize=(12, 0.01), dpi=500)
    plt.axis('off')
    plt.rc('font', size=6)  # Set font size
    plt.table(cellText=total_expected_points_combined.values, colLabels=total_expected_points_combined.columns,colLoc='left', cellLoc='left', loc='center')
    plt.savefig("total_expected_points_table.png", format="png", bbox_inches='tight')
    plt.close()
    pdf.image("total_expected_points_table.png", x=5, y=71, w=200)
    
    y_position = 100  # Initial y position after the table image
    pdf.set_xy(5, y_position)

    # Generate the DataFrame summary table for total expected points
    total_expected_points_combined_top_6 = total_expected_points_combined_top_6.round(2)
    total_expected_points_combined_top_6 = total_expected_points_combined_top_6.sort_values(by='Expected points per game', ascending=False)
    total_expected_points_combined_top_6 = total_expected_points_combined_top_6[total_expected_points_combined_top_6['team_name'].isin(['Kolding','Fredericia','Horsens','OB','Hvidovre','Esbjerg'])]
    plt.figure(figsize=(12, 0.01), dpi=500)
    plt.axis('off')
    plt.rc('font', size=6)  # Set font size
    plt.table(cellText=total_expected_points_combined_top_6.values, colLabels=total_expected_points_combined_top_6.columns,colLoc='left', cellLoc='left', loc='center')
    plt.savefig("total_expected_points_table_top_6.png", format="png", bbox_inches='tight')
    plt.close()
    pdf.image("total_expected_points_table_top_6.png", x=5, y=100, w=200)
    
    y_position = 130  # Initial y position after the table image
    pdf.set_xy(5, y_position)


    for position, df in position_dataframes.items():
        filtered_df = df[(df['team_name'] == 'Horsens')]
        if 'player_position' in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=['label', 'team_name', 'player_position', 'age_today'])
        else:
            filtered_df = filtered_df.drop(columns=['label', 'team_name', 'age_today'])

        if 'player_positionSide' in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=['player_positionSide'])

        numeric_columns = filtered_df.select_dtypes(include='number').columns.tolist()
        numeric_columns.remove('minsPlayed')
        
        # Sum minsPlayed and mean for the rest of the numeric columns
        aggregation_dict = {col: 'mean' for col in numeric_columns}
        aggregation_dict['minsPlayed'] = 'sum'
        
        filtered_df = filtered_df.groupby('playerName').agg(aggregation_dict).reset_index()
        filtered_df = filtered_df[filtered_df['minsPlayed'] > 400]
        filtered_df = filtered_df.round(2)
        filtered_df['Total score'] = filtered_df['Total score'].astype(float)        
        reordered_columns = ['playerName', 'minsPlayed'] + numeric_columns
        filtered_df = filtered_df[reordered_columns]
        filtered_df = filtered_df.sort_values('Total score',ascending=False)
        pdf.set_font("Arial", size=6)
        pdf.cell(190, 4, txt=f"Position Report: {position}", ln=True, align='C')
        pdf.ln(0)

        # Add table headers
        pdf.set_font("Arial", size=6)
        headers = filtered_df.columns
        col_width = 30  # Fixed width for all columns except the last one
        last_col_width = 15  # Width for the last column

        for header in headers[:-1]:
            pdf.cell(col_width, 4, txt=header, border=1)
        pdf.cell(last_col_width, 4, txt=headers[-1], border=1)
        pdf.ln(4)

        # Add table content
        for index, row in filtered_df.iterrows():
            total_score = row['Total score']
            if total_score < 4:
                fill_color = (255, 0, 0)  # Red
            elif 4 <= total_score <= 6:
                fill_color = (255, 255, 0)  # Yellow
            else:
                fill_color = (0, 255, 0)  # Green

            pdf.set_fill_color(*fill_color)

            # Add all cells for the row
            for value in row.values[:-1]:
                pdf.cell(col_width, 4, txt=str(value), border=1, fill=True)
            pdf.cell(last_col_width, 4,txt=str(row.values[-1]), border=1, fill=True)
            pdf.ln(4)  # Move to next line for the next player's row

        pdf.ln(0)  # Add some space between position reports

    pdf.output(f"Progress reports/Progress_report_{today}.pdf")
    print(f'{today} progress report created')

create_pdf_progress_report(horsens_df,total_expected_points_combined,total_expected_points_combined_top_6,position_dataframes)

folder_path = 'C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/'

# List all files in the folder
files = os.listdir(folder_path)

# Iterate over each file
for file in files:
    # Check if the file is a PNG file and not 'logo.png'
    if file.endswith(".png") and file != "Logo.png":
        # Construct the full path to the file
        file_path = os.path.join(folder_path, file)
        # Remove the file
        os.remove(file_path)
        print(f"Deleted: {file_path}")
