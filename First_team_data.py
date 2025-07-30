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
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path
import os
import requests
import glob


st.set_page_config(layout='wide')


@st.cache_data
def load_subs():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/subs%20DNK_1_Division_2025_2026.csv'
    df_subs = pd.read_csv(url)
    df_subs['label'] = (df_subs['label'] + ' ' + df_subs['date'])
    return df_subs   

@st.cache_data
def load_match_stats():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/matchstats_all%20DNK_1_Division_2025_2026.csv'
    match_stats = pd.read_csv(url)
    match_stats['label'] = (match_stats['label'] + ' ' + match_stats['date'])
    return match_stats

@st.cache_data
def load_possession_data():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/Horsens/Horsens_possession_data.csv'
    df_possession = pd.read_csv(url)
    #df_possession = pd.read_csv(r'C:\Users\Seamus-admin\Documents\GitHub\AC-Horsens-First-Team\DNK_1_Division_2025_2026\Horsens\Horsens_possession_data.csv')
    df_possession['label'] = (df_possession['label'] + ' ' + df_possession['date']).astype(str)
    return df_possession

@st.cache_data
def load_def_line_data():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/Horsens_Defensive_line_data.csv'
    def_line = pd.read_csv(url)
    
    return def_line

@st.cache_data
def load_possession_stats():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/possession_stats_all%20DNK_1_Division_2025_2026.csv'
    df_possession_stats = pd.read_csv(url)
    df_possession_stats['label'] = (df_possession_stats['label'] + ' ' + df_possession_stats['date'])
    return df_possession_stats

@st.cache_data
def load_xg():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/Horsens/Horsens_xg_data.csv'
    df_xg = pd.read_csv(url)
    df_xg['label'] = (df_xg['label'] + ' ' + df_xg['date'])
    df_xg['team_name'].str.replace(' ', '_')
    df_xg = df_xg[['playerName','label','team_name','x','y','321','periodId','timeMin','timeSec','9','24','25','26']]
    return df_xg

@st.cache_data
def load_all_xg():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/xg_all%20DNK_1_Division_2025_2026.csv'
    df_xg_all = pd.read_csv(url)
    df_xg_all['label'] = (df_xg_all['label'] + ' ' + df_xg_all['date'])
    df_xg_all['team_name'].str.replace(' ', '_')
    return df_xg_all

@st.cache_data
def load_pv():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/Horsens/Horsens_pv_data.csv'
    df_pv = pd.read_csv(url)
    df_pv['label'] = (df_pv['label'] + ' ' + df_pv['date'])
    df_pv['id'] = df_pv['id'].astype(str)
    df_pv['team_name'].str.replace(' ', '_')
    return df_pv

@st.cache_data
def load_xA():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/xA_all%20DNK_1_Division_2025_2026.csv'
    df_xA = pd.read_csv(url)
    df_xA['label'] = (df_xA['label'] + ' ' + df_xA['date']).astype(str)
    return df_xA

@st.cache_data
def load_ppda():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/ppda_all%20DNK_1_Division_2025_2026.csv'
    df_ppda = pd.read_csv(url)
    df_ppda['label'] = (df_ppda['label'] + ' ' + df_ppda['date']).astype(str)
    return df_ppda

@st.cache_data
def load_pv_all():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/pv_all%20DNK_1_Division_2025_2026.csv'
    df_pv_all = pd.read_csv(url)
    df_pv_all['label'] = (df_pv_all['label'] + ' ' + df_pv_all['date']).astype(str)
    return df_pv_all

@st.cache_data
def load_squads():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/squads%20DNK_1_Division_2025_2026.csv'
    squads = pd.read_csv(url)
    return squads

@st.cache_data
def load_physical_data():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/Physical%20data_all.csv'
    physical_data = pd.read_csv(url)
    return physical_data

@st.cache_data
def load_set_piece_data():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/set_piece_DNK_1_Division_2025_2026.csv'
    df_set_piece = pd.read_csv(url)
    df_set_piece['label'] = (df_set_piece['label'] + ' ' + df_set_piece['date']).astype(str)
    return df_set_piece

@st.cache_data
def load_transitions_data():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/Transitions%20DNK_1_Division_2025_2026.csv'
    df_transitions = pd.read_csv(url)
    df_transitions['label'] = (df_transitions['label'] + ' ' + df_transitions['date']).astype(str)
    return df_transitions

@st.cache_data
def load_on_ball_sequences():
    url = 'https://raw.githubusercontent.com/AC-Horsens/AC-Horsens-First-Team/main/DNK_1_Division_2025_2026/Horsens/Horsens_on_ball_sequences.csv'
    df_on_ball_sequences = pd.read_csv(url)
    df_on_ball_sequences['date'] = pd.to_datetime(df_on_ball_sequences['local_date'])
    df_on_ball_sequences['label'] = (
        df_on_ball_sequences['description'] + ' ' + df_on_ball_sequences['date'].astype(str)
    )
    return df_on_ball_sequences

def Process_data_spillere(df_xA,df_pv_all,df_match_stats,df_xg_all,squads):

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

    minutter_kamp = 0
    minutter_total = 300
        
    df_possession_xa = df_xA.rename(columns={'318.0': 'xA'})
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
    df_matchstats = df_match_stats[['player_matchName','player_playerId','contestantId','duelLost','aerialLost','player_position','player_positionSide','successfulOpenPlayPass','totalContest','duelWon','penAreaEntries','accurateBackZonePass','possWonDef3rd','wonContest','accurateFwdZonePass','openPlayPass','totalBackZonePass','minsPlayed','fwdPass','finalThirdEntries','ballRecovery','totalFwdZonePass','successfulFinalThirdPasses','totalFinalThirdPasses','attAssistOpenplay','aerialWon','totalAttAssist','possWonMid3rd','interception','totalCrossNocorner','interceptionWon','attOpenplay','touchesInOppBox','attemptsIbox','totalThroughBall','possWonAtt3rd','accurateCrossNocorner','bigChanceCreated','accurateThroughBall','totalLayoffs','accurateLayoffs','totalFastbreak','shotFastbreak','formationUsed','label','match_id','date','possLostAll','attemptsConcededIbox']]
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

    df_scouting[df_scouting.select_dtypes(include='number').columns] = \
    df_scouting.select_dtypes(include='number').fillna(0)    
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
    df_scouting['bigChanceCreated_per90'] = (df_scouting['bigChanceCreated'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['dribble %'] = (df_scouting['wonContest'].astype(float) / df_scouting['totalContest'].astype(float)) * 100
    df_scouting['dribble_per90'] = (df_scouting['wonContest'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['touches_in_box_per90'] = (df_scouting['touchesInOppBox'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['totalThroughBall_per90'] = (df_scouting['totalThroughBall'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['attemptsIbox_per90'] = (df_scouting['attemptsIbox'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['aerialWon_per90'] = (df_scouting['aerialWon'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['possLost_per90'] = (df_scouting['possLostAll'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['Attempts conceded in box per 90'] = (df_scouting['attemptsConcededIbox'].astype(float)/df_scouting['minsPlayed'].astype(float)) * 90

    df_scouting[df_scouting.select_dtypes(include='number').columns] = \
    df_scouting.select_dtypes(include='number').fillna(0)    

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

        return df_balanced_central_defender
  
    def fullbacks():
        df_backs = df_scouting[((df_scouting['player_position'] == 'Defender') | (df_scouting['player_position'] == 'Wing Back')| (df_scouting['player_position'] == 'Midfielder')) & 
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

        df_backs = df_backs.fillna(0)

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
        df_sekser = df_sekser.fillna(0)
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
        df_sekser = df_sekser.fillna(0)

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
        df_otter = df_scouting[
            (df_scouting['player_position'] == 'Midfielder') & 
            (df_scouting['player_positionSide'].isin(['Centre/Right', 'Left/Centre']))
        ]        
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
                [3 if row['Defending_'] < 5 else 1, 3 if row['Passing_'] < 5 else 1, 
                3 if row['Progressive_ball_movement'] < 5 else 1, 3 if row['Possession_value'] < 5 else 1]
            ), axis=1
        )

        # Prepare final output
        df_otter = df_otter.fillna(0)

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
            (
                (df_scouting['player_position'].isin(['Attacking Midfielder', 'Striker'])) &
                (df_scouting['player_positionSide'].isin(['Centre/Right', 'Left/Centre']))
            )
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
        df_10 = df_10.fillna(0)

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
        df_winger = df_winger.fillna(0)
        
        df_winger = df_winger[['playerName', 'team_name', 'age_today', 'minsPlayed', 'label', 
                    'Passing_', 'Chance_creation', 'Goalscoring_', 'Possession_value', 'Total score']]

        df_winger_total = df_winger[['playerName', 'team_name', 'minsPlayed', 
                                    'age_today', 'Passing_', 'Chance_creation', 'Goalscoring_', 'Possession_value', 'Total score']]
        df_winger_total = df_winger_total.groupby(['playerName', 'team_name', 'age_today']).mean().reset_index()
        minutter = df_winger.groupby(['playerName', 'team_name', 'age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_winger_total['minsPlayed total'] = minutter['minsPlayed']

        df_winger_total = df_winger_total[df_winger_total['minsPlayed total'].astype(int) >= minutter_total]
        df_winger_total = df_winger_total.sort_values('Total score', ascending=False)

        return df_winger

    def Classic_striker():
        df_striker = df_scouting[(df_scouting['player_position'] == 'Striker') & (df_scouting['player_positionSide']=='Centre')]
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
        df_striker = df_striker.fillna(0)
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
        'Wingback': fullbacks(),
        'Number 6' : number6(),
        'Number 8': number8(),
        'Number 10': number10(),
        'Winger': winger(),
        'Striker': Classic_striker(),
    }

def plot_heatmap_location(data):
    pitch = Pitch(pitch_type='opta', line_zorder=2, pitch_color='grass', line_color='white')
    fig, ax = pitch.draw(figsize=(6.6, 4.125))
    fig.set_facecolor('#22312b')
    bin_statistic = pitch.bin_statistic(data['x'], data['y'], statistic='count', bins=(50, 25))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot')
    st.pyplot(fig)

df_xA = load_xA()
df_pv = load_pv_all()
df_match_stats = load_match_stats()
df_xg_all = load_all_xg()
squads = load_squads()

position_dataframes = Process_data_spillere(df_xA, df_pv, df_match_stats, df_xg_all, squads)
balanced_central_defender_df = position_dataframes['Central defender']
fullbacks_df = position_dataframes['Wingback']
number6_df = position_dataframes['Number 6']
number8_df = position_dataframes['Number 8']
number10_df = position_dataframes['Number 10']
winger_df = position_dataframes['Winger']
classic_striker_df = position_dataframes['Striker']
def Dashboard():
    xml_files = glob.glob('DNK_1_Division_2025_2026/Horsens/XML files/*.xml')
    if xml_files:
        selected_xml = st.selectbox('Select an XML file to download:', xml_files)
        with open(selected_xml, 'rb') as f:
            st.download_button(
                label="Download selected XML",
                data=f,
                file_name=selected_xml,
                mime='application/xml'
            )
    else:
        st.write('No XML files found in this directory.')
    df_possession = load_possession_data()
    df_possession = df_possession[~df_possession['28.0'].astype(str).str.lower().eq('true')]

    df_set_pieces = load_set_piece_data()
    df_set_pieces = df_set_pieces[~df_set_pieces['28.0'].astype(str).str.lower().eq('true')]

    df_transitions = load_transitions_data()
    df_transitions = df_transitions[~df_transitions['28.0'].astype(str).str.lower().eq('true')]

    excluded_ids = df_set_pieces['id'].dropna().unique()

    # Remove transitions that are also in set pieces
    df_transitions = df_transitions[~df_transitions['id'].isin(excluded_ids)]
    excluded_transition_id = df_transitions['id'].dropna().unique()
    # Standardisér team_name og match_state
    for df in [df_possession, df_transitions, df_set_pieces]:
        df['team_name'] = df['team_name'].apply(lambda x: x if x == 'Horsens' else 'Opponent')
        df['match_state'] = df['match_state'].apply(lambda x: x if x == 'Horsens' or x == 'draw' else 'Opponent')

    df_possession['date'] = pd.to_datetime(df_possession['date'])
    df_possession = df_possession.sort_values(by='date', ascending=False)
    st.title('AC Horsens First Team Dashboard')
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
        option1 = st.checkbox('Horsens',True)
    with col2:
        option2 = st.checkbox('Draw',True)
    with col3:
        option3 = st.checkbox('Opponent',True)

    df_transitions = df_transitions[df_transitions['label'].isin(match_choice)]
    df_possession = df_possession[df_possession['label'].isin(match_choice)]
    df_set_pieces = df_set_pieces[df_set_pieces['label'].isin(match_choice)]
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
        game_state_df = game_state_df[game_state_df['match_state'].isin(['Horsens', 'draw', 'Opponent'])]
    # Case when two options are selected
    elif option1 and option2:
        game_state_df= game_state_df[game_state_df['match_state'].isin(['Horsens', 'draw'])]
    elif option1 and option3:
        game_state_df = game_state_df[game_state_df['match_state'].isin(['Horsens', 'Opponent'])]
    elif option2 and option3:
        game_state_df = game_state_df[game_state_df['match_state'].isin(['draw', 'Opponent'])]
    # Case when only one option is selected
    elif option1:
        game_state_df = game_state_df[game_state_df['match_state'] == 'Horsens']
    elif option2:
        game_state_df = game_state_df[game_state_df['match_state'] == 'draw']
    elif option3:
        game_state_df = game_state_df[game_state_df['match_state'] == 'Opponent']
    # Show
    game_state_df = game_state_df.groupby('match_state')['duration'].sum().reset_index()
    st.dataframe(game_state_df,hide_index=True)
    # Calculate passes per possession
    state_duration = game_state_df['duration'].sum()
    selected_states = []
    if option1:
        selected_states.append('Horsens')
    if option2:
        selected_states.append('draw')
    if option3:
        selected_states.append('Opponent')

    df_possession = df_possession[df_possession['match_state'].isin(selected_states)]
    df_transitions = df_transitions[df_transitions['match_state'].isin(selected_states)]
    df_set_pieces = df_set_pieces[df_set_pieces['match_state'].isin(selected_states)]


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
    teams = ['Horsens', 'Opponent']
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
    df_possession = df_possession[~df_possession['id'].isin(excluded_transition_id)]

    @st.cache_data
    def get_transition_summaries(df_transitions):
        goals = df_transitions[df_transitions['typeId'] == 16]
        goals_per_player = goals.groupby('playerName').size().reset_index(name='goals')

        summary = df_transitions.groupby(['playerName','team_name'])[['assist', 'sequence_xG', '321.0', '322.0']].sum().reset_index()
        summary = summary.rename(columns={'321.0':'xG','322.0': 'Post shot xG'})
        summary = summary.merge(goals_per_player, on='playerName', how='left')
        summary['goals'] = summary['goals'].fillna(0).astype(int)

        team_summary = summary.groupby('team_name')[['xG','Post shot xG','goals']].sum().reset_index().round(2)
        player_summary = summary.round(2)

        return team_summary, player_summary

    def get_breakthrough_summaries(df_possession):
        # Load set piece data
        df_possession['date'] = pd.to_datetime(df_possession['date'])
        df_set_pieces['date'] = pd.to_datetime(df_set_pieces['date'])
        df_transitions['date'] = pd.to_datetime(df_transitions['date'])

        excluded_ids = pd.concat([
            df_set_pieces[['id']],
            df_transitions[['id']]
        ])['id'].dropna().unique()

        # Filter df_possession to keep only rows not in the excluded_ids
        df_open_play = df_possession[~df_possession['id'].isin(excluded_ids)]

        # --- Continue as before ---
        goals = df_open_play[df_open_play['typeId'] == 16]
        goals_per_player = goals.groupby('playerName').size().reset_index(name='goals')

        summary = df_open_play.groupby(['playerName', 'team_name'])[['assist', 'sequence_xG', '321.0', '322.0']].sum().reset_index()
        summary = summary.rename(columns={'321.0': 'xG', '322.0': 'Post shot xG'})
        summary = summary.merge(goals_per_player, on='playerName', how='left')
        summary['goals'] = summary['goals'].fillna(0).astype(int)

        team_summary = summary.groupby('team_name')[['xG', 'Post shot xG', 'goals']].sum().reset_index().round(2)
        player_summary = summary.round(2)

        return team_summary, player_summary

    def plot_transitions(transitions_starts, vis_type):
        pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', line_zorder=2)
        fig, ax = pitch.draw(figsize=(10, 7))

        if vis_type == "Pitch Scatter":
            size_scale = transitions_starts['sequence_xG'].fillna(0) * 500
            pitch.scatter(
                transitions_starts['x'], transitions_starts['y'],
                ax=ax, s=size_scale, color='yellow', edgecolors='black', alpha=0.7
            )

            for _, row in transitions_starts.iterrows():
                pitch.annotate(
                    f"{row['playerName']} ({row['sequence_xG']:.2f})",
                    xy=(row['x'], row['y']),
                    ax=ax,
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    color='black'
                )

            st.pyplot(fig)

        elif vis_type == "Heatmap":
            plot_heatmap_location(transitions_starts)

    def Breakthrough():
        on_ball_sequences = load_on_ball_sequences()
        labels_df = df_possession[['match_id','date', 'label']].drop_duplicates()
        states_df = df_possession[['match_id','date','label', 'contestantId', 'timeMin', 'timeSec', 'match_state']]
        # Merge only on match_id to get label
        on_ball_sequences = on_ball_sequences.merge(labels_df, on=['match_id','date','label'], how='left')
        # Merge on full key to get match_state
        on_ball_sequences = on_ball_sequences.merge(states_df,left_on=['match_id','date','label', 'timemin_last', 'timesec_last'],right_on=['match_id','date','label', 'timeMin', 'timeSec'],how='left')
        on_ball_sequences = on_ball_sequences[on_ball_sequences['label'].isin(match_choice)]

        on_ball_sequences = on_ball_sequences.sort_values(['date', 'timemin_last', 'timesec_last'])
        on_ball_sequences = on_ball_sequences.ffill()
        mask_timeMin = on_ball_sequences['timeMin'].isna()
        on_ball_sequences.loc[mask_timeMin, 'timeMin'] = (
            0.5 * on_ball_sequences.loc[mask_timeMin, 'timemin_first'].astype(float) +
            0.5 * on_ball_sequences.loc[mask_timeMin, 'timemin_last'].astype(float)
        )

        # Similarly for timeSec
        mask_timeSec = on_ball_sequences['timeSec'].isna()
        on_ball_sequences.loc[mask_timeSec, 'timeSec'] = (
            0.5 * on_ball_sequences.loc[mask_timeSec, 'timesec_first'].astype(float) +
            0.5 * on_ball_sequences.loc[mask_timeSec, 'timesec_last'].astype(float)
        )
        on_ball_sequences = on_ball_sequences.drop(['date', 'timemin_last', 'timesec_last'], axis=1)
        on_ball_sequences['match_state'] = on_ball_sequences['match_state'].fillna('draw')
        on_ball_sequences = on_ball_sequences[on_ball_sequences['match_state'].isin(selected_states)]

        on_ball_sequences = on_ball_sequences[on_ball_sequences['poss_player_name'] != on_ball_sequences['receiver_name']]

        unique_sequences = on_ball_sequences.drop_duplicates(subset=['label', 'sequence_id'])

        counts = []
        for concept in ['High base', 'Width', 'Pocket']:
            # Filter rows where this concept is True
            concept_df = unique_sequences[unique_sequences[concept] == True]

            # Total sequences with this concept
            count_total = concept_df.shape[0]

            # Count how many sequences had a deep run opportunity
            deep_run_opportunity = concept_df['deep_run_opportunity'].sum()

            # Count how many sequences had at least one deep run (convert to binary)
            # Group by sequence and check if any deep_run == 1
            deep_run_per_sequence = (
                on_ball_sequences[on_ball_sequences[concept] == True]
                .groupby(['label', 'sequence_id'])['deep_run']
                .any()
                .sum()
            )

            counts.append({
                'Tactical Concept': concept,
                'Count': count_total,
                'Deep run opportunities': deep_run_opportunity,
                'Deep Runs': deep_run_per_sequence
            })

        # Format to DataFrame
        tactical_counts = pd.DataFrame(counts)

        # Display in Streamlit
        st.dataframe(tactical_counts, use_container_width=True, hide_index=True)
        
        assistzone = unique_sequences[unique_sequences['poss_in_assist_zone'] == True]

        # Compute metrics
        assistzone_count = assistzone.shape[0]
        avg_teammates_in_box = assistzone['num_teammates_in_box'].mean()

        # Format into a one-row DataFrame
        assistzone_summary = pd.DataFrame([{
            'Assist Zone Sequences': assistzone_count,
            'Avg Teammates in Box': round(avg_teammates_in_box, 2)
        }])

        # Display in Streamlit
        st.dataframe(assistzone_summary, use_container_width=True, hide_index=True)

        tactical_concepts = ['High base', 'Width', 'Pocket']
        selected_concept = st.selectbox("Choose tactical concept to analyze:", tactical_concepts)

        st.subheader(f'Analysis for: {selected_concept}')

        # Filter to only unique sequences for this concept
        concept_df = on_ball_sequences.copy()
        concept_df = concept_df[concept_df[selected_concept] == True]

        # ==== OPTIONS BETWEEN LINES ====
        options_count = (
            concept_df.groupby(['match_id','label', 'sequence_id','timeMin'])['option_between_lines'].sum()
            .reset_index()
            .rename(columns={'option_between_lines': 'options_between_lines_count'})
        )

        summary = options_count.copy()
        summary['ten_min_bin'] = (summary['timeMin'] // 10) * 10

        options_per_10min = (
            summary.groupby(['match_id', 'label', 'ten_min_bin'])['options_between_lines_count']
            .mean()
            .reset_index()
        )

        st.markdown('**Options between lines per situation**')
        for match in options_per_10min['label'].unique():
            match_data = options_per_10min[options_per_10min['label'] == match]
            fig = px.line(
                match_data,
                x='ten_min_bin',
                y='options_between_lines_count',
                range_y=[0, 5],
                title=f"{selected_concept} – Match {match}",
                labels={'ten_min_bin': 'Minute (10-min bin)', 'options_between_lines_count': 'Avg Options Between Lines'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # ==== DEEP RUN CONVERSIONS ====
        concept_df['ten_min_bin'] = (concept_df['timeMin'] // 10) * 10

        sequence_opportunity = (
            concept_df.groupby(['match_id', 'label', 'ten_min_bin', 'sequence_id'])['deep_run_opportunity'].any()
            .reset_index()
            .rename(columns={'deep_run_opportunity': 'has_deep_run_opportunity'})
        )

        sequence_deep_runs = (
            concept_df.groupby(['match_id', 'label', 'ten_min_bin', 'sequence_id'])['deep_run'].sum()
            .reset_index()
            .rename(columns={'deep_run': 'deep_runs'})
        )

        seq_summary = sequence_opportunity.merge(
            sequence_deep_runs, on=['match_id', 'label', 'ten_min_bin', 'sequence_id']
        )
        seq_summary = seq_summary[seq_summary['has_deep_run_opportunity']]

        deep_run_binned = (
            seq_summary.groupby(['match_id', 'label', 'ten_min_bin'])
            .agg(
                deep_run_opportunities=('sequence_id', 'nunique'),
                deep_runs=('deep_runs', 'sum')
            )
            .reset_index()
        )

        deep_run_binned['conversion_rate'] = deep_run_binned.apply(
            lambda row: row['deep_runs'] / row['deep_run_opportunities'] if row['deep_run_opportunities'] > 0 else None,
            axis=1
        )

        st.markdown('**Deep run conversion rate**')
        for match in deep_run_binned['label'].unique():
            match_data = deep_run_binned[deep_run_binned['label'] == match]
            fig = px.line(
                match_data,
                x='ten_min_bin',
                y='conversion_rate',
                range_y=[0, 1],
                title=f"{selected_concept} – Match {match}",
                labels={'ten_min_bin': 'Minute (10-min bin)', 'conversion_rate': 'Deep Run Conversion Rate'}
            )
            st.plotly_chart(fig, use_container_width=True)


        zone1_mask = (
            ((df_possession['x'] >= 66) & (df_possession['x'] <= 80) & (df_possession['y'] >= 40) & (df_possession['y'] <= 60)) |
            ((df_possession['x'] > 80) & (df_possession['y'] >= 63) & (df_possession['y'] <= 83)) |
            ((df_possession['x'] > 80) & (df_possession['y'] >= 17) & (df_possession['y'] <= 37))
        )

        assist_zone_possessions = df_possession[zone1_mask]

        # --- Count assist zone actions ---
        assist_zone_counts = assist_zone_possessions.groupby('team_name').size().reset_index(name='Assist zone actions')

        horsens_az = assist_zone_counts.loc[assist_zone_counts['team_name'] == 'Horsens', 'Assist zone actions'].sum()
        opponent_az = assist_zone_counts.loc[assist_zone_counts['team_name'] != 'Horsens', 'Assist zone actions'].sum()

        # --- Create base difference_df ---
        difference_df = pd.DataFrame({
            'Team': ['Horsens', 'Opponents'],
            'Assist zone actions': [horsens_az, opponent_az],
            'AZ difference': [horsens_az - opponent_az, opponent_az - horsens_az]
        })

        # --- Danger zone polygon setup ---
        danger_zone_poly = [
            (100, 44),     # Right goalpost
            (100, 56),     # Left goalpost
            (85, 62.5),    # Top outer
            (85, 37.5)     # Bottom outer
        ]

        danger_path = Path(danger_zone_poly)
        positions = df_possession[['x', 'y']].to_numpy()
        inside_dangerzone = danger_path.contains_points(positions)
        dangerzone_actions = df_possession[inside_dangerzone]

        # --- Count danger zone actions ---
        dangerzone_counts = dangerzone_actions.groupby('team_name').size().reset_index(name='Dangerzone actions')

        horsens_dz = dangerzone_counts.loc[dangerzone_counts['team_name'] == 'Horsens', 'Dangerzone actions'].sum()
        opponent_dz = dangerzone_counts.loc[dangerzone_counts['team_name'] != 'Horsens', 'Dangerzone actions'].sum()

        # --- Create danger zone df and merge ---
        dangerzone_df = pd.DataFrame({
            'Team': ['Horsens', 'Opponents'],
            'Dangerzone actions': [horsens_dz, opponent_dz],
            'DZ difference': [horsens_dz - opponent_dz, opponent_dz - horsens_dz]
        })

        # --- Merge assist and danger zone dataframes ---
        full_df = difference_df.merge(dangerzone_df, on='Team', how='outer')

        # --- Show summary table ---
        st.subheader("Assist Zone & Dangerzone Action Summary")
        st.dataframe(full_df, hide_index=True)

        # --- Draw pitch with assist + danger zones ---
        pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', half=True)
        fig, ax = pitch.draw(figsize=(6, 9))

        # Draw assist zones
        assist_zones = [
            {'label': 'Assistzone', 'x': 66, 'y': 33, 'width': 14, 'height': 35, 'color': 'yellow'},
            {'label': 'Assistzone', 'x': 83, 'y': 63, 'width': 17, 'height': 20, 'color': 'yellow'},
            {'label': 'Assistzone', 'x': 83, 'y': 17, 'width': 17, 'height': 20, 'color': 'yellow'}
        ]

        for zone in assist_zones:
            rect = Rectangle(
                (zone['x'], zone['y']),
                zone['width'],
                zone['height'],
                edgecolor='black',
                facecolor=zone['color'],
                alpha=0.4,
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(
                zone['x'] + zone['width'] / 2,
                zone['y'] + zone['height'] / 2,
                zone['label'],
                ha='center', va='center',
                fontsize=10,
                color='black',
                weight='bold'
            )

        # Draw trapezoidal danger zone
        danger_polygon = Polygon(
            danger_zone_poly,
            closed=True,
            edgecolor='black',
            facecolor='red',
            alpha=0.4,
            linewidth=2
        )
        ax.add_patch(danger_polygon)

        # Add label for dangerzone
        ax.text(
            92, 50,
            'Dangerzone',
            ha='center', va='center',
            fontsize=10,
            color='black',
            weight='bold'
        )

        # Display the full figure
        st.pyplot(fig)

        team_summary, player_summary = get_breakthrough_summaries(df_possession)

        st.subheader("Team Breakthrough Summary")
        st.dataframe(team_summary, hide_index=True)

        st.subheader("Player Breakthrough Summary")
        horsens_summary = player_summary[player_summary['team_name'] == 'Horsens']
        st.dataframe(horsens_summary.sort_values(['goals','xG'], ascending=False), hide_index=True)

    def transitions():
        zone1_mask = (
            ((df_transitions['x'] >= 66) & (df_transitions['x'] <= 80) & (df_transitions['y'] >= 40) & (df_transitions['y'] <= 60)) |
            ((df_transitions['x'] > 83) & (df_transitions['y'] >= 63) & (df_transitions['y'] <= 83)) |
            ((df_transitions['x'] > 83) & (df_transitions['y'] >= 17) & (df_transitions['y'] <= 37))
        )

        assist_zone_possessions = df_transitions[zone1_mask]
        # --- Count assist zone actions ---
        assist_zone_counts = assist_zone_possessions.groupby('team_name').size().reset_index(name='Assist zone actions')

        horsens_az = assist_zone_counts.loc[assist_zone_counts['team_name'] == 'Horsens', 'Assist zone actions'].sum()
        opponent_az = assist_zone_counts.loc[assist_zone_counts['team_name'] != 'Horsens', 'Assist zone actions'].sum()

        # --- Create base difference_df ---
        difference_df = pd.DataFrame({
            'Team': ['Horsens', 'Opponents'],
            'Transition Assist zone actions': [horsens_az, opponent_az],
            'Transition AZ difference': [horsens_az - opponent_az, opponent_az - horsens_az]
        })

        # --- Danger zone polygon setup ---
        danger_zone_poly = [
            (100, 44),     # Right goalpost
            (100, 56),     # Left goalpost
            (85, 62.5),    # Top outer
            (85, 37.5)     # Bottom outer
        ]

        danger_path = Path(danger_zone_poly)
        positions = df_transitions[['x', 'y']].to_numpy()
        inside_dangerzone = danger_path.contains_points(positions)
        dangerzone_actions = df_transitions[inside_dangerzone]

        # --- Count danger zone actions ---
        dangerzone_counts = dangerzone_actions.groupby('team_name').size().reset_index(name='Dangerzone actions')

        horsens_dz = dangerzone_counts.loc[dangerzone_counts['team_name'] == 'Horsens', 'Dangerzone actions'].sum()
        opponent_dz = dangerzone_counts.loc[dangerzone_counts['team_name'] != 'Horsens', 'Dangerzone actions'].sum()

        # --- Create danger zone df and merge ---
        dangerzone_df = pd.DataFrame({
            'Team': ['Horsens', 'Opponents'],
            'Transition Dangerzone actions': [horsens_dz, opponent_dz],
            'Transition DZ difference': [horsens_dz - opponent_dz, opponent_dz - horsens_dz]
        })

        # --- Merge assist and danger zone dataframes ---
        full_df = difference_df.merge(dangerzone_df, on='Team', how='outer')

        # --- Show summary table ---
        st.subheader("Assist Zone & Dangerzone Action Summary")
        st.dataframe(full_df, hide_index=True)

        # --- Draw pitch with assist + danger zones ---
        pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', half=True)
        fig, ax = pitch.draw(figsize=(6, 9))

        # Draw assist zones
        assist_zones = [
            {'label': 'Assistzone', 'x': 66, 'y': 33, 'width': 14, 'height': 35, 'color': 'yellow'},
            {'label': 'Assistzone', 'x': 83, 'y': 63, 'width': 17, 'height': 20, 'color': 'yellow'},
            {'label': 'Assistzone', 'x': 83, 'y': 17, 'width': 17, 'height': 20, 'color': 'yellow'}
        ]

        for zone in assist_zones:
            rect = Rectangle(
                (zone['x'], zone['y']),
                zone['width'],
                zone['height'],
                edgecolor='black',
                facecolor=zone['color'],
                alpha=0.4,
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(
                zone['x'] + zone['width'] / 2,
                zone['y'] + zone['height'] / 2,
                zone['label'],
                ha='center', va='center',
                fontsize=10,
                color='black',
                weight='bold'
            )

        # Draw trapezoidal danger zone
        danger_polygon = Polygon(
            danger_zone_poly,
            closed=True,
            edgecolor='black',
            facecolor='red',
            alpha=0.4,
            linewidth=2
        )
        ax.add_patch(danger_polygon)

        # Add label for dangerzone
        ax.text(
            92, 50,
            'Dangerzone',
            ha='center', va='center',
            fontsize=10,
            color='black',
            weight='bold'
        )

        # Display the full figure
        st.pyplot(fig)

        # Load and show cached summary stats
        team_summary, player_summary = get_transition_summaries(df_transitions)

        st.subheader("Team Offensive transitions Summary")
        st.dataframe(team_summary, hide_index=True)

        st.subheader("Player Offensive transitions Summary")
        horsens_summary = player_summary[player_summary['team_name'] == 'Horsens']
        st.dataframe(horsens_summary.sort_values(['goals','xG'], ascending=False), hide_index=True)

        # Transition starts visualization (only reruns below here on selectbox change)
        st.subheader('Transition starts with shot')

        transitions_starts = df_transitions[
            (df_transitions['possession_index'] == 1) & 
            (df_transitions['team_name'] == 'Horsens') &
            (df_transitions['sequence_duration'] > 0) &
            (df_transitions['sequence_xG'] > 0)
        ]

        vis_type = st.selectbox("Choose visualization type", ["Pitch Scatter", "Heatmap"])
        plot_transitions(transitions_starts, vis_type)

        # Central corridor analysis
        central_corridor_mask = (
            (transitions_starts['x'] >= 30) & (transitions_starts['x'] <= 70) &
            (transitions_starts['y'] >= 20) & (transitions_starts['y'] <= 80)
        )
        central_transitions = transitions_starts[central_corridor_mask]
        total_transitions = len(transitions_starts)
        central_count = len(central_transitions)
        percentage_central = (central_count / total_transitions * 100) if total_transitions > 0 else 0

        st.markdown(f"**Transitions startet in transition start zone:** {central_count} out of {total_transitions} "
                    f"({percentage_central:.1f}%)")

    def Buildup():
        on_ball_sequences = load_on_ball_sequences()
        labels_df = df_possession[['match_id','date', 'label']].drop_duplicates()
        states_df = df_possession[['match_id','date','label', 'contestantId', 'timeMin', 'timeSec', 'match_state']]
        # Merge only on match_id to get label
        on_ball_sequences = on_ball_sequences.merge(labels_df, on=['match_id','date','label'], how='left')
        # Merge on full key to get match_state
        on_ball_sequences = on_ball_sequences.merge(states_df,left_on=['match_id','date','label', 'timemin_last', 'timesec_last'],right_on=['match_id','date','label', 'timeMin', 'timeSec'],how='left')
        on_ball_sequences = on_ball_sequences[on_ball_sequences['label'].isin(match_choice)]

        on_ball_sequences = on_ball_sequences.sort_values(['date', 'timemin_last', 'timesec_last'])
        on_ball_sequences = on_ball_sequences.ffill()
        mask_timeMin = on_ball_sequences['timeMin'].isna()
        on_ball_sequences.loc[mask_timeMin, 'timeMin'] = (
            0.5 * on_ball_sequences.loc[mask_timeMin, 'timemin_first'].astype(float) +
            0.5 * on_ball_sequences.loc[mask_timeMin, 'timemin_last'].astype(float)
        )

        # Similarly for timeSec
        mask_timeSec = on_ball_sequences['timeSec'].isna()
        on_ball_sequences.loc[mask_timeSec, 'timeSec'] = (
            0.5 * on_ball_sequences.loc[mask_timeSec, 'timesec_first'].astype(float) +
            0.5 * on_ball_sequences.loc[mask_timeSec, 'timesec_last'].astype(float)
        )
        on_ball_sequences = on_ball_sequences.drop(['date', 'timemin_last', 'timesec_last'], axis=1)
        on_ball_sequences['match_state'] = on_ball_sequences['match_state'].fillna('draw')
        on_ball_sequences = on_ball_sequences[on_ball_sequences['match_state'].isin(selected_states)]

        on_ball_sequences = on_ball_sequences[on_ball_sequences['poss_player_name'] != on_ball_sequences['receiver_name']]

        filtered_df = on_ball_sequences[on_ball_sequences['Low base'] == True]
        low_base_count = filtered_df.drop_duplicates(subset=['sequence_id', 'label'])

        low_base_count = low_base_count.sort_values(['label', 'sequence_id']).reset_index(drop=True)

        # Shift to compare current row with previous row
        prev_label = low_base_count['label'].shift()
        prev_seq_id = low_base_count['sequence_id'].shift()

        # Keep rows where label changes OR sequence_id is not +1
        non_consecutive = (low_base_count['label'] != prev_label) | \
                        (low_base_count['sequence_id'] != prev_seq_id + 1)

        # Filter
        filtered_single_instances = low_base_count[non_consecutive]

        # Display
        st.write(f'Low base situations with time: {len(filtered_single_instances)}')

        # For each sequence, does any receiver have time_on_ball True? (use the original df, not filtered)
        seq_has_time_on = (
            filtered_df.groupby(['match_id','label', 'sequence_id','timeMin'])['time_on_ball'].any()
            .reset_index()
            .rename(columns={'time_on_ball': 'has_time_on_ball'})
        )

        # Count option_between_lines Trues per sequence/match (for has_opp_behind == False only)
        options_count = (
            filtered_df.groupby(['match_id','label', 'sequence_id','timeMin'])['option_between_lines'].sum()
            .reset_index()
            .rename(columns={'option_between_lines': 'options_between_lines_count'})
        )

        summary = options_count
        summary['ten_min_bin'] = (summary['timeMin'] // 10) * 10

        # Group and get mean
        options_per_5min = (
            summary.groupby(['match_id', 'label', 'ten_min_bin'])['options_between_lines_count']
            .mean()
            .reset_index()
        )

        import plotly.express as px
        st.header('Options between lines per low base situation')
        for match in options_per_5min['label'].unique():
            match_data = options_per_5min[options_per_5min['label'] == match]
            fig = px.line(
                match_data,
                x='ten_min_bin',
                y='options_between_lines_count',
                range_y = [0,5],
                title=f"Match {match}",
                labels={'ten_min_bin': 'Minute (10-min bin)', 'options_between_lines_count': 'Avg Options Between Lines'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        on_ball_sequences['ten_min_bin'] = (on_ball_sequences['timeMin'] // 10) * 10

        sequence_opportunity = (
            on_ball_sequences.groupby(['match_id', 'label', 'ten_min_bin', 'sequence_id'])['deep_run_opportunity'].any()
            .reset_index()
            .rename(columns={'deep_run_opportunity': 'has_deep_run_opportunity'})
        )

        # Numerator: How many deep_runs (receivers) in the sequence/bin/match?
        sequence_deep_runs = (
            on_ball_sequences.groupby(['match_id', 'label', 'ten_min_bin', 'sequence_id'])['deep_run'].sum()
            .reset_index()
            .rename(columns={'deep_run': 'deep_runs'})
        )

        # Merge and keep only sequences with opportunity
        seq_summary = sequence_opportunity.merge(sequence_deep_runs, on=['match_id','label','ten_min_bin','sequence_id'])
        seq_summary = seq_summary[seq_summary['has_deep_run_opportunity']]

        # Now group by match/bin to get totals
        deep_run_binned = (
            seq_summary.groupby(['match_id', 'label', 'ten_min_bin'])
            .agg(
                deep_run_opportunities=('sequence_id', 'nunique'),  # 1 per sequence with opportunity
                deep_runs=('deep_runs', 'sum')
            )
            .reset_index()
        )

        deep_run_binned['conversion_rate'] = deep_run_binned.apply(
            lambda row: row['deep_runs'] / row['deep_run_opportunities'] if row['deep_run_opportunities'] > 0 else None,
            axis=1
        )

        import plotly.express as px
        st.header('Deep runs per opportunity')

        for match in deep_run_binned['label'].unique():
            match_data = deep_run_binned[deep_run_binned['label'] == match]
            fig = px.line(
                match_data,
                x='ten_min_bin',
                y='conversion_rate',
                range_y=[0, 1],  # Conversion rate from 0 to 1
                title=f"Match {match}",
                labels={'ten_min_bin': 'Minute (10-min bin)', 'conversion_rate': 'Deep Run Conversion Rate'}
            )
            st.plotly_chart(fig, use_container_width=True)

    def Defending():
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

        def_line = load_def_line_data()
        labels_df = df_possession[['match_id','date', 'label']].drop_duplicates()
        states_df = df_possession[['match_id','date','label', 'contestantId', 'timeMin', 'timeSec', 'match_state']]

        # Merge only on match_id to get label
        def_line = def_line.merge(labels_df, on='match_id', how='left')

        # Merge on full key to get match_state
        def_line = def_line.merge(states_df, on=['match_id','date','label', 'contestantId', 'timeMin', 'timeSec'], how='left')
        def_line = def_line.sort_values(['date','timeMin','timeSec'])
        def_line = def_line.ffill()


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
            yaxis_range=[70, 100],  # Set y-axis limits
        )

        st.plotly_chart(fig, use_container_width=True)

        # Define assist zone masks
        zone1_mask = (
            ((df_possession['x'] >= 66) & (df_possession['x'] <= 80) & (df_possession['y'] >= 40) & (df_possession['y'] <= 60)) |
            ((df_possession['x'] > 83) & (df_possession['y'] >= 63) & (df_possession['y'] <= 83)) |
            ((df_possession['x'] > 83) & (df_possession['y'] >= 17) & (df_possession['y'] <= 37))
        )

        assist_zone_possessions = df_possession[zone1_mask]

        # --- Count assist zone actions ---
        assist_zone_counts = assist_zone_possessions.groupby('team_name').size().reset_index(name='Assist zone actions')

        horsens_az = assist_zone_counts.loc[assist_zone_counts['team_name'] == 'Horsens', 'Assist zone actions'].sum()
        opponent_az = assist_zone_counts.loc[assist_zone_counts['team_name'] != 'Horsens', 'Assist zone actions'].sum()

        # --- Create base difference_df ---
        difference_df = pd.DataFrame({
            'Team': ['Horsens', 'Opponents'],
            'Assist zone actions': [horsens_az, opponent_az],
            'AZ difference': [horsens_az - opponent_az, opponent_az - horsens_az]
        })

        # --- Danger zone polygon setup ---
        danger_zone_poly = [
            (100, 44),     # Right goalpost
            (100, 56),     # Left goalpost
            (85, 62.5),    # Top outer
            (85, 37.5)     # Bottom outer
        ]

        danger_path = Path(danger_zone_poly)
        positions = df_possession[['x', 'y']].to_numpy()
        inside_dangerzone = danger_path.contains_points(positions)
        dangerzone_actions = df_possession[inside_dangerzone]

        # --- Count danger zone actions ---
        dangerzone_counts = dangerzone_actions.groupby('team_name').size().reset_index(name='Dangerzone actions')

        horsens_dz = dangerzone_counts.loc[dangerzone_counts['team_name'] == 'Horsens', 'Dangerzone actions'].sum()
        opponent_dz = dangerzone_counts.loc[dangerzone_counts['team_name'] != 'Horsens', 'Dangerzone actions'].sum()

        # --- Create danger zone df and merge ---
        dangerzone_df = pd.DataFrame({
            'Team': ['Horsens', 'Opponents'],
            'Dangerzone actions': [horsens_dz, opponent_dz],
            'DZ difference': [horsens_dz - opponent_dz, opponent_dz - horsens_dz]
        })

        # --- Merge assist and danger zone dataframes ---
        full_df = difference_df.merge(dangerzone_df, on='Team', how='outer')

        # --- Show summary table ---
        st.subheader("Assist Zone & Dangerzone Action Summary")
        st.dataframe(full_df, hide_index=True)

        # --- Draw pitch with assist + danger zones ---
        pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white', half=True)
        fig, ax = pitch.draw(figsize=(6, 9))

        # Draw assist zones
        assist_zones = [
            {'label': 'Assistzone', 'x': 66, 'y': 33, 'width': 14, 'height': 35, 'color': 'yellow'},
            {'label': 'Assistzone', 'x': 83, 'y': 63, 'width': 17, 'height': 20, 'color': 'yellow'},
            {'label': 'Assistzone', 'x': 83, 'y': 17, 'width': 17, 'height': 20, 'color': 'yellow'}
        ]

        for zone in assist_zones:
            rect = Rectangle(
                (zone['x'], zone['y']),
                zone['width'],
                zone['height'],
                edgecolor='black',
                facecolor=zone['color'],
                alpha=0.4,
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(
                zone['x'] + zone['width'] / 2,
                zone['y'] + zone['height'] / 2,
                zone['label'],
                ha='center', va='center',
                fontsize=10,
                color='black',
                weight='bold'
            )

        # Draw trapezoidal danger zone
        danger_polygon = Polygon(
            danger_zone_poly,
            closed=True,
            edgecolor='black',
            facecolor='red',
            alpha=0.4,
            linewidth=2
        )
        ax.add_patch(danger_polygon)

        # Add label for dangerzone
        ax.text(
            92, 50,
            'Dangerzone',
            ha='center', va='center',
            fontsize=10,
            color='black',
            weight='bold'
        )

        # Display the full figure
        st.pyplot(fig)

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
        'Build up':Buildup,
        'Breakthrough':Breakthrough,
        'Defending': Defending,
        'Set pieces': set_pieces,
        'Transitions': transitions
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

def Opposition_analysis():
    
    balanced_central_defender_df = position_dataframes['Central defender']
    balanced_central_defender_df['label'] = balanced_central_defender_df['label'].str.replace(' ', '_')
    
    fullbacks_df = position_dataframes['Wingback']
    fullbacks_df['label'] = fullbacks_df['label'].str.replace(' ', '_')

    number6_df = position_dataframes['Number 6']
    number6_df['label'] = number6_df['label'].str.replace(' ', '_')

    number8_df = position_dataframes['Number 8']
    number8_df['label'] = number8_df['label'].str.replace(' ', '_')

    number10_df = position_dataframes['Number 10']
    number10_df['label'] = number10_df['label'].str.replace(' ', '_')

    winger_df = position_dataframes['Winger']
    winger_df['label'] = winger_df['label'].str.replace(' ', '_')

    classic_striker_df = position_dataframes['Striker']
    classic_striker_df['label'] = classic_striker_df['label'].str.replace(' ', '_')

    matchstats_df = load_match_stats()
    matchstats_df = matchstats_df.rename(columns={'player_matchName': 'playerName'})
    matchstats_df = matchstats_df.groupby(['contestantId','team_name','label', 'date']).sum().reset_index()
    matchstats_df['label'] = np.where(matchstats_df['label'].notnull(), 1, matchstats_df['label'])
    date_format = '%Y-%m-%d'
    matchstats_df['date'] = pd.to_datetime(matchstats_df['date'], format=date_format)
    min_date = matchstats_df['date'].min()
    max_date = matchstats_df['date'].max()

    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    date_options = date_range.strftime(date_format)  # Convert dates to the specified format

    default_end_date = date_options[-1]

    default_end_date_dt = pd.to_datetime(default_end_date, format=date_format)
    default_start_date_dt = default_end_date_dt - pd.Timedelta(days=2)  # Subtract 14 days
    default_start_date = default_start_date_dt.strftime(date_format)  # Convert to string

    # Set the default start and end date values for the select_slider
    selected_start_date, selected_end_date = st.select_slider(
        'Choose dates',
        options=date_options,
        value=(min_date.strftime(date_format), max_date.strftime(date_format))
    )
    
    selected_start_date = pd.to_datetime(selected_start_date, format=date_format)
    selected_end_date = pd.to_datetime(selected_end_date, format=date_format)
    filtered_data = matchstats_df[
        (matchstats_df['date'] >= selected_start_date) & (matchstats_df['date'] <= selected_end_date)
    ]    
    
    xg_df = load_all_xg()

    # Filter for xG
    xg_df_openplay = xg_df[xg_df['321'] > 0]

    # Aggregate xG by team and match (contestantId, team_name, label, date)
    xg_df_openplay = (
        xg_df_openplay.groupby(['contestantId', 'team_name', 'label', 'date'])['321']
        .sum()
        .reset_index()
        .rename(columns={'321': 'xG'})
    )

    # Ensure the date column is in datetime format
    xg_df_openplay['date'] = pd.to_datetime(xg_df_openplay['date'])

    # Calculate total xG for each match (grouping by label and date)
    match_total_xg = (
        xg_df_openplay.groupby(['label', 'date'])['xG']
        .sum()
        .reset_index()
        .rename(columns={'xG': 'total match xG'})
    )

    # Merge the total match xG into the team-level data
    xg_df_openplay = xg_df_openplay.merge(match_total_xg, on=['label', 'date'], how='left')

    # Calculate xG against for each team
    xg_df_openplay['xG against'] = xg_df_openplay['total match xG'] - xg_df_openplay['xG']

    # Optional: Drop intermediate columns if needed
    xg_df_openplay = xg_df_openplay.drop(columns=['total match xG'])
    xg_df_openplay['label'] = np.where(xg_df_openplay['label'].notnull(), 1, xg_df_openplay['label'])

    df_ppda = load_ppda()
    df_ppda = df_ppda.groupby(['team_name','date']).sum().reset_index()
    df_ppda['date'] = pd.to_datetime(df_ppda['date'])
    df_ppda['PPDA'] = df_ppda['PPDA'].astype(float).round(2)
    df_ppda = df_ppda[['team_name','date', 'PPDA']]
    matchstats_df = xg_df_openplay.merge(filtered_data,on=['contestantId','label','team_name','date'])
    matchstats_df = df_ppda.merge(matchstats_df)

    matchstats_df = matchstats_df.drop(columns='date')
    # Perform aggregation
    matchstats_df = matchstats_df.groupby(['contestantId', 'team_name']).agg({
        'label': 'sum',  # Example of a column to sum
        'penAreaEntries': 'sum',  # Example of another column to sum
        'xG': 'sum',
        'xG against' : 'sum',  # Example of a column to average
        'duelLost': 'sum',
        'duelWon': 'sum',
        'openPlayPass': 'sum',
        'successfulOpenPlayPass': 'sum',
        'accurateBackZonePass': 'sum',
        'totalBackZonePass': 'sum',
        'accurateFwdZonePass': 'sum',
        'totalFwdZonePass': 'sum',
        'possWonDef3rd': 'sum',
        'possWonMid3rd': 'sum',
        'possWonAtt3rd': 'sum',
        'fwdPass': 'sum',
        'finalThirdEntries': 'sum',
        'successfulFinalThirdPasses': 'sum',
        'totalFinalThirdPasses': 'sum',
        'attAssistOpenplay': 'sum',
        'totalAttAssist': 'sum',
        'totalCrossNocorner': 'sum',
        'accurateCrossNocorner': 'sum',
        'totalLongBalls': 'sum',
        'PPDA': 'mean',
        }).reset_index()
    matchstats_df = matchstats_df.round(2)

    matchstats_df = matchstats_df.rename(columns={'label': 'matches'})
    matchstats_df['PenAreaEntries per match'] = matchstats_df['penAreaEntries'] / matchstats_df['matches']
    matchstats_df['xG per match'] = matchstats_df['xG'] / matchstats_df['matches']
    matchstats_df['xG against per match'] = matchstats_df['xG against'] / matchstats_df['matches']
    matchstats_df['Duels per match'] = (matchstats_df['duelLost'] + matchstats_df['duelWon']) /matchstats_df['matches']
    matchstats_df['Duels won %'] = (matchstats_df['duelWon'] / (matchstats_df['duelWon'] + matchstats_df['duelLost']))*100	
    matchstats_df['Passes per game'] = matchstats_df['openPlayPass'] / matchstats_df['matches']
    matchstats_df['Pass accuracy %'] = (matchstats_df['successfulOpenPlayPass'] / matchstats_df['openPlayPass'])*100
    matchstats_df['Back zone pass accuracy %'] = (matchstats_df['accurateBackZonePass'] / matchstats_df['totalBackZonePass'])*100
    matchstats_df['Forward zone pass accuracy %'] = (matchstats_df['accurateFwdZonePass'] / matchstats_df['totalFwdZonePass'])*100
    matchstats_df['possWonDef3rd %'] = (matchstats_df['possWonDef3rd'] / (matchstats_df['possWonDef3rd'] + matchstats_df['possWonMid3rd'] + matchstats_df['possWonAtt3rd']))*100    
    matchstats_df['possWonMid3rd %'] = (matchstats_df['possWonMid3rd'] / (matchstats_df['possWonDef3rd'] + matchstats_df['possWonMid3rd'] + matchstats_df['possWonAtt3rd']))*100    
    matchstats_df['possWonAtt3rd %'] = (matchstats_df['possWonAtt3rd'] / (matchstats_df['possWonDef3rd'] + matchstats_df['possWonMid3rd'] + matchstats_df['possWonAtt3rd']))*100   
    matchstats_df['Forward pass share %'] = (matchstats_df['fwdPass'] / matchstats_df['openPlayPass'])*100
    matchstats_df['Final third entries per match'] = matchstats_df['finalThirdEntries'] / matchstats_df['matches']
    matchstats_df['Final third pass accuracy %'] = (matchstats_df['successfulFinalThirdPasses'] / matchstats_df['totalFinalThirdPasses'])*100
    matchstats_df['Open play shot assists share'] = (matchstats_df['attAssistOpenplay'] / matchstats_df['totalAttAssist'])*100
    matchstats_df['Long pass share %'] = (matchstats_df['totalLongBalls'] / matchstats_df['openPlayPass'])*100
    matchstats_df['Crosses'] = matchstats_df['totalCrossNocorner']
    matchstats_df['Cross accuracy %'] = (matchstats_df['accurateCrossNocorner'] / matchstats_df['totalCrossNocorner'])*100
    matchstats_df['PPDA per match'] = matchstats_df['PPDA']
    matchstats_df = matchstats_df[['team_name','matches','PenAreaEntries per match','xG per match','xG against per match','Duels per match','Duels won %','Passes per game','Pass accuracy %','Back zone pass accuracy %','Forward zone pass accuracy %','possWonDef3rd %','possWonMid3rd %','possWonAtt3rd %','Forward pass share %','Final third entries per match','Final third pass accuracy %','Open play shot assists share','PPDA per match','Long pass share %','Crosses','Cross accuracy %']]
    matchstats_df['team_name'] = matchstats_df['team_name'].str.replace(' ', '_')
    matchstats_df = matchstats_df.round(2)

    cols_to_rank = matchstats_df.drop(columns=['team_name']).columns
    ranked_df = matchstats_df.copy()
    for col in cols_to_rank:
        if col == 'PPDA per match':
            ranked_df[col + '_rank'] = matchstats_df[col].rank(axis=0, ascending=True)
        else:
            ranked_df[col + '_rank'] = matchstats_df[col].rank(axis=0, ascending=False)

    matchstats_df = ranked_df.merge(matchstats_df)
    matchstats_df = matchstats_df.round(2)

    matchstats_df = matchstats_df.set_index('team_name')
    matchstats_df = matchstats_df.drop(columns=['matches_rank'])
    matchstats_df = matchstats_df.apply(pd.to_numeric, errors='ignore')
    matchstats_df = matchstats_df.round(2)
    st.dataframe(matchstats_df)
    matchstats_df = matchstats_df.reset_index()

    # Select a team
    sorted_teams = matchstats_df['team_name'].sort_values()
    selected_team = st.selectbox('Choose team', sorted_teams)
    team_df = matchstats_df.loc[matchstats_df['team_name'] == selected_team]

    # Target ranks
    target_ranks = [1,1.5, 2,2.5, 3,3.5, 4,4.5, 9,9.5, 10,10.5, 11,11.5, 12,12.5]

    # Filter the selected team's ranks and values
    filtered_data_df = pd.DataFrame()
    col1, col2 = st.columns([1, 2])
    for col in team_df.columns:
        if col.endswith('_rank'):
            original_col = col[:-5]
            if any(team_df[col].isin(target_ranks)):
                filtered_ranks = team_df.loc[team_df[col].isin(target_ranks), col]
                filtered_values = team_df.loc[team_df[col].isin(target_ranks), original_col]
                filtered_data_df[original_col + '_rank'] = filtered_ranks.values
                filtered_data_df[original_col + '_value'] = filtered_values.values

    with col1:
        filtered_data_df = filtered_data_df.T
        st.dataframe(filtered_data_df)

    # ------------------------------------
    # Similar Teams Section (Moved up)
    # Find similar teams
    selected_columns = [
        'Duels per match_rank', 'Duels won %_rank', 'Passes per game_rank', 'Pass accuracy %_rank', 'possWonDef3rd %_rank',
        'possWonMid3rd %_rank', 'possWonAtt3rd %_rank', 'Forward pass share %_rank', 'Open play shot assists share_rank',
        'PPDA per match_rank', 'Long pass share %_rank', 'Crosses_rank', 'Cross accuracy %_rank'
    ]

    rank_columns = [col for col in matchstats_df.columns if col.endswith('_rank')]
    weighted_columns = ['Passes per game_rank', 'Long pass share %_rank', 'Forward pass share %_rank', 'PPDA per match_rank']

    # Update similarity score calculation by giving specific columns a weight of 3
    matchstats_df['similarity_score'] = matchstats_df.apply(
        lambda row: sum(
            (3 if rank_col in weighted_columns else 1) * abs(row[rank_col] - team_df[rank_col].values[0])
            for rank_col in selected_columns if not pd.isna(row[rank_col])
        ),
        axis=1
    )

    similar_teams = matchstats_df[matchstats_df['team_name'] != selected_team]
    similar_teams = similar_teams[similar_teams['team_name'] != 'Horsens']
    top_3_similar_teams = similar_teams.nsmallest(3, 'similarity_score')
    top_3_similar_teams = top_3_similar_teams.sort_values(by='similarity_score', ascending=False)

    with col2:
        st.write("Teams similar to the selected team:")
        st.dataframe(top_3_similar_teams[['team_name'] + rank_columns + ['similarity_score']], hide_index=True)


    st.header('Central defenders')    # ------------------------------------

    balanced_central_defender_df = balanced_central_defender_df[balanced_central_defender_df['team_name'] == 'Horsens']
    balanced_central_defender_df['match_date'] = pd.to_datetime(balanced_central_defender_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    today = datetime.now()
    three_months_ago = today - timedelta(days=60)

    # Filter for matches within the last two months
    balanced_central_defender_df = balanced_central_defender_df[
        (balanced_central_defender_df['match_date'] >= three_months_ago) & 
        (balanced_central_defender_df['match_date'] <= today)
    ]

    # Ensure 'match_date' column is not null
    balanced_central_defender_df = balanced_central_defender_df.dropna(subset=['match_date'])
    unique_dates = balanced_central_defender_df['match_date'].unique()

    # Get the latest 3 match dates
    latest_dates = pd.Series(unique_dates).nlargest(3)
    # Filter for rows with the latest match dates
    recent_matches_df = balanced_central_defender_df[balanced_central_defender_df['match_date'].isin(latest_dates)]

    # Filter for rows where 'label' contains one of the top 3 similar teams
    teams_list = top_3_similar_teams['team_name'].tolist()
    matches_with_teams_df = balanced_central_defender_df[balanced_central_defender_df['label'].str.contains('|'.join(teams_list))]

    # Ensure that matches involving the selected team are also considered in the similarity score
    matches_against_selected_team_df = balanced_central_defender_df[balanced_central_defender_df['label'].str.contains(selected_team)]

    # Combine recent matches, matches involving top teams, and matches against the selected team
    combined_df = pd.merge(recent_matches_df, matches_with_teams_df, how='outer')
    combined_df = pd.merge(combined_df, matches_against_selected_team_df, how='outer')

    # Sort and clean up combined data
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date','player_position'])
    st.dataframe(combined_df,hide_index = True)
    combined_df = combined_df.drop(columns = ['label'])
    # Aggregate and sort by 'Total score    '
    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}
    }).reset_index()

    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    agg_df = agg_df.round(2)
    st.dataframe(agg_df, hide_index=True)
    central_defender_df = agg_df.copy()
    st.header('Wingback')

    # Filter the fullbacks data for the 'Horsens' team
    fullbacks_df = fullbacks_df[fullbacks_df['team_name'] == 'Horsens']

    # Extract and convert 'match_date' from the 'label' column, dropping null values
    fullbacks_df['match_date'] = pd.to_datetime(fullbacks_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    fullbacks_df = fullbacks_df.dropna(subset=['match_date'])
    fullbacks_df = fullbacks_df[
        (fullbacks_df['match_date'] >= three_months_ago) & 
        (fullbacks_df['match_date'] <= today)
    ]
    # Use previously calculated 'latest_dates' to filter recent matches
    recent_matches_df = fullbacks_df[fullbacks_df['match_date'].isin(latest_dates)]

    # Filter for matches where 'label' contains one of the top 3 similar teams
    matches_with_teams_df = fullbacks_df[fullbacks_df['label'].str.contains('|'.join(teams_list))]
    matches_against_selected_team_df = fullbacks_df[fullbacks_df['label'].str.contains(selected_team)]

    # Combine recent matches and matches involving similar teams
    combined_df = pd.concat([
        recent_matches_df,
        matches_with_teams_df,
        matches_against_selected_team_df
    ], ignore_index=True).drop_duplicates()
    # Sort by 'Total score' and drop unnecessary columns
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date', 'player_position', 'player_positionSide'])
    combined_df = combined_df.round(2)
    # Display the combined DataFrame in Streamlit
    st.dataframe(combined_df, hide_index=True)

    # Drop the 'label' column for aggregation
    combined_df = combined_df.drop(columns=['label'])

    # Group by player and team, aggregating with sum and mean as needed
    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',  # Sum up the minutes played
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}  # Mean for other columns
    }).reset_index()

    # Sort by 'Total score' in descending order
    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    agg_df = agg_df.round(2)

    # Display the aggregated DataFrame in Streamlit
    st.dataframe(agg_df, hide_index=True)
    fullback_df = agg_df.copy()


    st.header('Number 6')
    fullbacks_df = number6_df[number6_df['team_name'] == 'Horsens']
    # Extract and convert 'match_date' from the 'label' column, dropping null values
    fullbacks_df['match_date'] = pd.to_datetime(fullbacks_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    fullbacks_df = fullbacks_df.dropna(subset=['match_date'])
    fullbacks_df = fullbacks_df[
        (fullbacks_df['match_date'] >= three_months_ago) & 
        (fullbacks_df['match_date'] <= today)
    ]
    # Use previously calculated 'latest_dates' to filter recent matches
    recent_matches_df = fullbacks_df[fullbacks_df['match_date'].isin(latest_dates)]

    # Filter for matches where 'label' contains one of the top 3 similar teams
    matches_with_teams_df = fullbacks_df[fullbacks_df['label'].str.contains('|'.join(teams_list))]
    matches_against_selected_team_df = fullbacks_df[fullbacks_df['label'].str.contains(selected_team)]

    # Combine recent matches and matches involving similar teams
    combined_df = pd.merge(recent_matches_df, matches_with_teams_df, how='outer')
    combined_df = pd.merge(combined_df, matches_against_selected_team_df, how='outer')

    # Sort by 'Total score' and drop unnecessary columns
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date', 'player_position'])
    combined_df = combined_df.round(2)

    # Display the combined DataFrame in Streamlit
    st.dataframe(combined_df, hide_index=True)

    # Drop the 'label' column for aggregation
    combined_df = combined_df.drop(columns=['label'])

    # Group by player and team, aggregating with sum and mean as needed
    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',  # Sum up the minutes played
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}  # Mean for other columns
    }).reset_index()

    # Sort by 'Total score' in descending order
    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    agg_df = agg_df.round(2)

    # Display the aggregated DataFrame in Streamlit
    st.dataframe(agg_df, hide_index=True)
    Number_6_df = agg_df.copy()

    st.header('Number 8')
    fullbacks_df = number8_df[number8_df['team_name'] == 'Horsens']
    fullbacks_df['match_date'] = pd.to_datetime(fullbacks_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    fullbacks_df = fullbacks_df.dropna(subset=['match_date'])
    fullbacks_df = fullbacks_df[
        (fullbacks_df['match_date'] >= three_months_ago) & 
        (fullbacks_df['match_date'] <= today)
    ]
    # Use previously calculated 'latest_dates' to filter recent matches
    recent_matches_df = fullbacks_df[fullbacks_df['match_date'].isin(latest_dates)]

    # Filter for matches where 'label' contains one of the top 3 similar teams
    matches_with_teams_df = fullbacks_df[fullbacks_df['label'].str.contains('|'.join(teams_list))]
    matches_against_selected_team_df = fullbacks_df[fullbacks_df['label'].str.contains(selected_team)]

    # Combine recent matches and matches involving similar teams
    combined_df = pd.merge(recent_matches_df, matches_with_teams_df, how='outer')
    combined_df = pd.merge(combined_df, matches_against_selected_team_df, how='outer')

    # Sort by 'Total score' and drop unnecessary columns
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date', 'player_position'])
    combined_df = combined_df.round(2)

    # Display the combined DataFrame in Streamlit
    st.dataframe(combined_df, hide_index=True)

    # Drop the 'label' column for aggregation
    combined_df = combined_df.drop(columns=['label'])

    # Group by player and team, aggregating with sum and mean as needed
    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',  # Sum up the minutes played
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}  # Mean for other columns
    }).reset_index()

    # Sort by 'Total score' in descending order
    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    agg_df = agg_df.round(2)

    # Display the aggregated DataFrame in Streamlit
    st.dataframe(agg_df, hide_index=True)
    Number_8_df = agg_df.copy()

    st.header('Number 10')
    fullbacks_df = number10_df[number10_df['team_name'] == 'Horsens']
    fullbacks_df['match_date'] = pd.to_datetime(fullbacks_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    fullbacks_df = fullbacks_df.dropna(subset=['match_date'])
    fullbacks_df = fullbacks_df[
        (fullbacks_df['match_date'] >= three_months_ago) & 
        (fullbacks_df['match_date'] <= today)
    ]
    # Use previously calculated 'latest_dates' to filter recent matches
    recent_matches_df = fullbacks_df[fullbacks_df['match_date'].isin(latest_dates)]

    # Filter for matches where 'label' contains one of the top 3 similar teams
    matches_with_teams_df = fullbacks_df[fullbacks_df['label'].str.contains('|'.join(teams_list))]
    matches_against_selected_team_df = fullbacks_df[fullbacks_df['label'].str.contains(selected_team)]

    # Combine recent matches and matches involving similar teams
    combined_df = pd.merge(recent_matches_df, matches_with_teams_df, how='outer')
    combined_df = pd.merge(combined_df, matches_against_selected_team_df, how='outer')

    # Sort by 'Total score' and drop unnecessary columns
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date'])
    combined_df = combined_df.round(2)

    # Display the combined DataFrame in Streamlit
    st.dataframe(combined_df, hide_index=True)

    # Drop the 'label' column for aggregation
    combined_df = combined_df.drop(columns=['label'])

    # Group by player and team, aggregating with sum and mean as needed
    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',  # Sum up the minutes played
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}  # Mean for other columns
    }).reset_index()

    # Sort by 'Total score' in descending order
    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    agg_df = agg_df.round(2)

    # Display the aggregated DataFrame in Streamlit
    st.dataframe(agg_df, hide_index=True)
    Number_10_df = agg_df.copy()

    st.header('Winger')
    fullbacks_df = winger_df[winger_df['team_name'] == 'Horsens']
    fullbacks_df['match_date'] = pd.to_datetime(fullbacks_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    fullbacks_df = fullbacks_df.dropna(subset=['match_date'])
    fullbacks_df = fullbacks_df[
        (fullbacks_df['match_date'] >= three_months_ago) & 
        (fullbacks_df['match_date'] <= today)
    ]
    # Use previously calculated 'latest_dates' to filter recent matches
    recent_matches_df = fullbacks_df[fullbacks_df['match_date'].isin(latest_dates)]

    # Filter for matches where 'label' contains one of the top 3 similar teams
    matches_with_teams_df = fullbacks_df[fullbacks_df['label'].str.contains('|'.join(teams_list))]
    matches_against_selected_team_df = fullbacks_df[fullbacks_df['label'].str.contains(selected_team)]

    # Combine recent matches and matches involving similar teams
    combined_df = pd.merge(recent_matches_df, matches_with_teams_df, how='outer')
    combined_df = pd.merge(combined_df, matches_against_selected_team_df, how='outer')

    # Sort by 'Total score' and drop unnecessary columns
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date'])
    combined_df = combined_df.round(2)

    # Display the combined DataFrame in Streamlit
    st.dataframe(combined_df, hide_index=True)

    # Drop the 'label' column for aggregation
    combined_df = combined_df.drop(columns=['label'])

    # Group by player and team, aggregating with sum and mean as needed
    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',  # Sum up the minutes played
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}  # Mean for other columns
    }).reset_index()

    # Sort by 'Total score' in descending order
    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    agg_df = agg_df.round(2)

    # Display the aggregated DataFrame in Streamlit
    st.dataframe(agg_df, hide_index=True)
    Winger_df = agg_df.copy()

    
    st.header('Striker')
    fullbacks_df = classic_striker_df[classic_striker_df['team_name'] == 'Horsens']
    fullbacks_df['match_date'] = pd.to_datetime(fullbacks_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    fullbacks_df = fullbacks_df.dropna(subset=['match_date'])
    fullbacks_df = fullbacks_df[
        (fullbacks_df['match_date'] >= three_months_ago) & 
        (fullbacks_df['match_date'] <= today)
    ]
    # Use previously calculated 'latest_dates' to filter recent matches
    recent_matches_df = fullbacks_df[fullbacks_df['match_date'].isin(latest_dates)]

    # Filter for matches where 'label' contains one of the top 3 similar teams
    matches_with_teams_df = fullbacks_df[fullbacks_df['label'].str.contains('|'.join(teams_list))]
    matches_against_selected_team_df = fullbacks_df[fullbacks_df['label'].str.contains(selected_team)]

    # Combine recent matches and matches involving similar teams
    combined_df = pd.merge(recent_matches_df, matches_with_teams_df, how='outer')
    combined_df = pd.merge(combined_df, matches_against_selected_team_df, how='outer')

    # Sort by 'Total score' and drop unnecessary columns
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date'])
    combined_df = combined_df.round(2)

    # Display the combined DataFrame in Streamlit
    st.dataframe(combined_df, hide_index=True)

    # Drop the 'label' column for aggregation
    combined_df = combined_df.drop(columns=['label'])

    # Group by player and team, aggregating with sum and mean as needed
    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',  # Sum up the minutes played
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}  # Mean for other columns
    }).reset_index()

    # Sort by 'Total score' in descending order
    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    agg_df = agg_df.round(2)

    # Display the aggregated DataFrame in Streamlit
    st.dataframe(agg_df, hide_index=True)
    Striker_df = agg_df.copy()

    def draw_pitch(position_data):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 70)
        ax.set_aspect('equal', adjustable='box')

        # Draw pitch outline and features
        plt.plot([0, 0, 100, 100, 0], [0, 70, 70, 0, 0], color="black", linewidth=2)  # Pitch outline
        plt.plot([50, 50], [0, 70], color="black", linewidth=1)  # Center line
        plt.plot([0, 16.5, 16.5, 0], [25, 25, 45, 45], color="black", linewidth=1)  # Left penalty area
        plt.plot([100, 83.5, 83.5, 100], [25, 25, 45, 45], color="black", linewidth=1)  # Right penalty area
        center_circle = plt.Circle((50, 35), 9.15, color="black", fill=False, linewidth=1)
        ax.add_patch(center_circle)

        # Define player positions on the pitch
        positions = {
            "Central Defenders": [(10, 35)],
            "Wingback": [(30, 55), (30, 15)],
            "Number 6": [(30, 35)],
            "Number 8": [(65, 45)],
            "Number 10": [(75, 25)],
            "Wingers": [(60, 60), (60, 10)],
            "Strikers": [(90, 35)],
        }

        # Add players to the positions
        for position, coords in positions.items():
            players = position_data.get(position, [])
            for x, y in coords:
                # Add position title
                plt.text(x, y + 3, position, fontsize=10, ha="center", color="blue", fontweight="bold")
                # Display the same list of players for both locations
                for j, (player_name, total_score) in enumerate(players):
                    offset_y = y - j * 2  # Adjust vertical offset for each player
                    plt.text(x, offset_y, f"{player_name} ({total_score})", fontsize=8, ha="center", color="black")

        # Remove axes for a clean look
        ax.axis('off')
        return fig

    # Prepare the data for positions
    position_data = {
        "Central Defenders": central_defender_df[['playerName', 'Total score']].values.tolist(),
        "Wingback": fullback_df[['playerName', 'Total score']].values.tolist(),
        "Number 6": Number_6_df[['playerName', 'Total score']].values.tolist(),
        "Number 8": Number_8_df[['playerName', 'Total score']].values.tolist(),
        "Number 10": Number_10_df[['playerName', 'Total score']].values.tolist(),
        "Wingers": Winger_df[['playerName', 'Total score']].values.tolist(),
        "Strikers": Striker_df[['playerName', 'Total score']].values.tolist(),
    }

    # Draw and display the pitch
    st.header("Depth chart on data")
    pitch_fig = draw_pitch(position_data)
    st.pyplot(pitch_fig)

    st.header('Set pieces')

    # Load the set pieces data
    df_set_pieces = load_set_piece_data()
    df_set_pieces['team_name'] = df_set_pieces['team_name'].str.replace(" ", "_")
    df_set_pieces = df_set_pieces[df_set_pieces['team_name'] == selected_team]
    df_set_pieces['date'] = pd.to_datetime(df_set_pieces['date'], format='%Y-%m-%d')
    df_set_pieces = df_set_pieces[
        (df_set_pieces['date'] >= selected_start_date) & (df_set_pieces['date'] <= selected_end_date)
    ]
    def preprocess_short_corners(df):
        """
        Preprocess the dataframe to set 223.0, 224.0, and 225.0 to False when 212.0 is less than 10 meters 
        and 6.0 is either True (boolean) or 'true' (string). Additionally, a new column 'short' is added, 
        which is True if the short corner criteria are met.
        """
        # Add 'short' column, True if 212.0 < 10 and 6.0 is True or 'true'
        df['short'] = (df['212.0'] < 20) & ((df['6.0'] == True) | (df['6.0'] == 'true'))
        
        # Set 223.0, 224.0, and 225.0 to False if the corner is classified as 'short'
        df.loc[df['short'], ['223.0', '224.0', '225.0']] = False
        
        return df

    # Apply preprocessing to the set pieces data
    df_set_pieces = preprocess_short_corners(df_set_pieces)
    # Function to get the first contact and finisher for each possession

    def get_first_contact_and_finisher(df):
        result = []

        # Iterate over each unique possession and label combination
        for possession_id, group in df.groupby(['possessionId', 'label']):
            group = group.sort_values('set_piece_index')  # Sort by set_piece_index (time order)

            # Identify the row where 6.0 is True (corner taker)
            corner_taker_row = group[group['6.0'] == True]

            if not corner_taker_row.empty:
                corner_taker_index = corner_taker_row.index[0]
                label = corner_taker_row['label'].values[0]  # Get the label for xG aggregation

                # Check if the outcome of the corner taker is 0 (indicating failed corner)
                outcome = corner_taker_row['outcome'].values[0]

                # First contact logic based on outcome
                if outcome == 0:
                    first_contact_player = 'opponent'  # If outcome is 0, first contact is by the opponent
                else:
                    # First contact: Always the player from the next event after the corner taker
                    next_event_row = group[group.index > corner_taker_index].head(1)
                    if not next_event_row.empty:
                        first_contact_player = next_event_row['playerName'].values[0]
                    else:
                        first_contact_player = None

                # Finisher: The last player in the possession
                finisher_player = group.iloc[-1]['playerName']
                
                # xG for the possession (sum of all xG for the same possessionId and label)
                possession_xg = group['321.0'].sum()

                # Determine the type of corner for the entire possession
                inswinger = group['223.0'].any()  # Inswinger if any 223.0 is True within the possession
                outswinger = group['224.0'].any()  # Outswinger if any 224.0 is True within the possession
                straight = group['225.0'].any()  # Straight if any 225.0 is True within the possession
                short = group['short'].any()  # Short corner if any 'short' is True within the possession

                result.append({
                    'possessionId': str(possession_id),  # Convert possessionId to string
                    'first_contact_player': first_contact_player,
                    'finisher_player': finisher_player,
                    'xg': possession_xg,
                    'label': str(label),  # Convert label to string
                    'inswinger': inswinger,
                    'outswinger': outswinger,
                    'straight': straight,
                    'short': short
                })

        return pd.DataFrame(result)

    # Function to plot heatmaps for first contact using mplsoccer's VerticalPitch
    def plot_heatmap(df, title):
        pitch = VerticalPitch(pitch_type='opta', half=True, line_zorder=2, pitch_color='grass', line_color='white')
        fig, ax = pitch.draw()

        # Extract coordinates based on available data
        x_coords = df['140.0']  # x-coordinate column
        y_coords = df['141.0']  # y-coordinate column

        # Generate heatmap based on x and y coordinates
        bin_statistic = pitch.bin_statistic(x_coords, y_coords, statistic='count', bins=(50, 50))  # Adjust bins if needed
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='none')

        # Set plot title
        ax.set_title(title)

        # Display the heatmap in Streamlit
        st.pyplot(fig)

    # Function to split data for left and right side based on where the corner was taken
    def split_by_side(df, corner_type_column):
        """
        Split the dataframe into right side and left side based on where the corner was taken.
        For short corners, use the 'short' column to determine if a corner is short.
        """
        # Right side: y < 30 for the corner-taking row
        right_side = df[(df['6.0'] == True) & (df['y'] < 30) & (df[corner_type_column] == True)]
        
        # Left side: y > 70 for the corner-taking row
        left_side = df[(df['6.0'] == True) & (df['y'] > 70) & (df[corner_type_column] == True)]
        
        return right_side, left_side

    # Apply the function to inswingers (223.0)
    right_inswingers, left_inswingers = split_by_side(df_set_pieces, '223.0')

    # Apply the function to outswingers (224.0)
    right_outswingers, left_outswingers = split_by_side(df_set_pieces, '224.0')

    # Apply the function to short corners (using the new 'short' column)
    def split_by_side_for_short_corners(df):
        """
        Special handling for short corners. Split based on the 'short' column (whether short is True).
        """
        # Right side: y < 30 for the corner-taking row
        right_side = df[(df['6.0'] == True) & (df['y'] < 30) & (df['short'] == True)]
        
        # Left side: y > 70 for the corner-taking row
        left_side = df[(df['6.0'] == True) & (df['y'] > 70) & (df['short'] == True)]
        
        return right_side, left_side

    # Apply the function to short corners
    right_shorts, left_shorts = split_by_side_for_short_corners(df_set_pieces)
    col1, col2 = st.columns(2)

    # Display heatmaps for each type in the correct columns
    with col1:
        # Left side heatmaps
        st.subheader('First Contact - Left Side')
        plot_heatmap(left_inswingers, "First Contact - Inswingers (Left Side)")
        plot_heatmap(left_outswingers, "First Contact - Outswingers (Left Side)")
        plot_heatmap(left_shorts, "First Contact - Short Corners (Left Side)")

    with col2:
        # Right side heatmaps
        st.subheader('First Contact - Right Side')
        plot_heatmap(right_inswingers, "First Contact - Inswingers (Right Side)")
        plot_heatmap(right_outswingers, "First Contact - Outswingers (Right Side)")
        plot_heatmap(right_shorts, "First Contact - Short Corners (Right Side)")

    def summarize_xg_by_player(df, player_column, xg_column):
        """
        Summarize the total xG for players based on their role (first contact or finisher).
        Group by the player and aggregate the total xG.
        """
        summary = df.groupby(player_column)[xg_column].sum().reset_index()
        summary = summary[summary[xg_column] > 0]  # Filter out players with 0 xG
        return summary.sort_values(by=xg_column, ascending=False)

    # Create the summary for first contact and finisher for each corner type
    def summarize_first_contact_and_finisher(df, corner_type):
        """
        Summarize the first contact and finisher for each player based on corner type.
        """
        # Filter based on corner type (inswingers, outswingers, short)
        df_corner_type = df[df[corner_type] == True]
        
        # Summarize first contact
        first_contact_summary = summarize_xg_by_player(df_corner_type, 'first_contact_player', 'xg')
        first_contact_summary = first_contact_summary.rename(columns={'xg': f'first_contact_xg_{corner_type}'})

        # Summarize finisher
        finisher_summary = summarize_xg_by_player(df_corner_type, 'finisher_player', 'xg')
        finisher_summary = finisher_summary.rename(columns={'xg': f'finisher_xg_{corner_type}'})

        return first_contact_summary, finisher_summary
    first_contact_finisher_df = get_first_contact_and_finisher(df_set_pieces)
    def enforce_exclusivity(row):
        # If 'short' is True, set all others to False
        if row['short']:
            row['inswinger'] = False
            row['outswinger'] = False
            row['straight'] = False
        # Else, if 'inswinger' is True, set others to False
        elif row['inswinger']:
            row['outswinger'] = False
            row['straight'] = False
            row['short'] = False
        # Else, if 'outswinger' is True, set the remaining to False
        elif row['outswinger']:
            row['straight'] = False
            row['short'] = False
        # 'straight' is the lowest priority, so no further changes are needed if only it is True
        return row    # Summarize first contact and finisher for each corner type (inswingers, outswingers, shorts)
    first_contact_finisher_df = first_contact_finisher_df = first_contact_finisher_df.apply(enforce_exclusivity, axis=1)
    first_contact_inswingers, finisher_inswingers = summarize_first_contact_and_finisher(first_contact_finisher_df, 'inswinger')
    first_contact_outswingers, finisher_outswingers = summarize_first_contact_and_finisher(first_contact_finisher_df, 'outswinger')
    first_contact_shorts, finisher_shorts = summarize_first_contact_and_finisher(first_contact_finisher_df, 'short')

    # Display the results in Streamlit or a summary table
    st.header('xG Summary by Player for First Contact and Finisher')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('Inswingers')
        st.write('First Contact - Inswingers')
        st.dataframe(first_contact_inswingers, hide_index=True)

        st.write('Finisher - Inswingers')
        st.dataframe(finisher_inswingers, hide_index=True)

    with col2:
        st.subheader('Outswingers')
        st.write('First Contact - Outswingers')
        st.dataframe(first_contact_outswingers, hide_index=True)
        st.write('Finisher - Outswingers')
        st.dataframe(finisher_outswingers, hide_index=True)

    with col3:
        st.subheader('Short Corners')
        st.write('First Contact - Short Corners')
        st.dataframe(first_contact_shorts, hide_index=True)
        st.write('Finisher - Short Corners')
        st.dataframe(finisher_shorts, hide_index=True)

    def calculate_total_xg_by_corner_type(finisher_inswingers, finisher_outswingers, finisher_shorts):
        # Calculate the total xG for each corner type by summing the xG column in each dataframe
        total_inswinger_xg = finisher_inswingers['finisher_xg_inswinger'].sum()
        total_outswinger_xg = finisher_outswingers['finisher_xg_outswinger'].sum()
        total_short_xg = finisher_shorts['finisher_xg_short'].sum()

        # Create a summary dataframe
        total_xg_summary = pd.DataFrame({
            'Corner Type': ['Inswinger', 'Outswinger', 'Short'],
            'Total xG': [total_inswinger_xg, total_outswinger_xg, total_short_xg]
        })
        
        # Optionally, add a row for the overall total xG
        total_xg_summary.loc['Total'] = ['All Types', total_xg_summary['Total xG'].sum()]
        
        return total_xg_summary

    # Calculate the total xG by corner type
    total_xg_summary = calculate_total_xg_by_corner_type(finisher_inswingers, finisher_outswingers, finisher_shorts)

    # Display the total xG summary
    st.subheader('Total xG by Corner Type Across All Players')
    st.dataframe(total_xg_summary, hide_index=True)

def Physical_data():
    df = load_physical_data()
    df_matchstats = load_match_stats()
    df_matchstats = df_matchstats[['player_matchName','minsPlayed','player_playerId','contestantId','label','match_id','date']]
    df_matchstats = df_matchstats.rename(columns={'player_playerId': 'optaUuid', 'match_id': 'Opta match id'})
    df = df.merge(df_matchstats,on=['Opta match id','optaUuid'])

    total_df = df[['Team','label','High Speed Running Distance','High Speed Running Count','Sprinting Count','Sprinting Distance','Total Distance']]
    total_df = total_df.groupby(['Team','label']).sum().reset_index()
    total_df = total_df[['Team','High Speed Running Distance','High Speed Running Count','Sprinting Count','Sprinting Distance','Total Distance']]
    total_df = total_df.groupby('Team').mean().reset_index()
    total_df = total_df.round(2)

    df = df[df['minsPlayed'].astype(int) > 30]
    df = df[['Player','Team','label','minsPlayed','High Speed Running Distance','High Speed Running Count','Sprinting Count','Sprinting Distance','Total Distance']]
    
    metric_columns = ['High Speed Running Distance', 'High Speed Running Count', 'Sprinting Count', 
                      'Sprinting Distance', 'Total Distance']
    
    # Adjust each metric column to be per 90 minutes
    for col in metric_columns:
        df[col] = (df[col].astype(float) / df['minsPlayed'].astype(float)) * 90
    df = df.round(2)
    team = sorted(df['Team'].unique())
    col1,col2 = st.columns(2)
    with col1:
        teams = st.selectbox('Choose team',team)
    team_df = df[df['Team'] == teams]
    matches = team_df['label'].unique()
    with col2:
        match = st.multiselect('Choose match',matches,default=matches)

    team_df = team_df[team_df['label'].isin(match)]

    team_df = team_df[['Player','minsPlayed','High Speed Running Distance','High Speed Running Count','Sprinting Count','Sprinting Distance','Total Distance']]
    sum_df = df[['Player','minsPlayed']]
    sum_df = sum_df.groupby('Player').sum().reset_index()
    sum_df = sum_df[sum_df['minsPlayed'] > 300]
    df = df.merge(sum_df,on='Player',how='inner')
    df = df[['Player','Team','High Speed Running Distance','High Speed Running Count','Sprinting Count','Sprinting Distance','Total Distance']]

    df = df.groupby(['Player','Team']).mean().reset_index()
    df = df.round(2)

    st.write('All matches')
    st.dataframe(df,hide_index=True)
    team_df = team_df.merge(sum_df,on='Player',how='inner')
    team_df = team_df[['Player','High Speed Running Distance','High Speed Running Count','Sprinting Count','Sprinting Distance','Total Distance']]

    team_df = team_df.groupby(['Player']).mean().reset_index()
    team_df = team_df.round(2)

    st.write('Chosen matches')
    st.dataframe(team_df, hide_index=True)

    st.write("Team Total Metrics (Sorted by Metric)")
    for metric in metric_columns:
        # Sort data by the metric in descending order
        sorted_df = total_df[['Team', metric]].sort_values(by=metric, ascending=False)
        
        # Create a bar chart with Plotly
        fig = go.Figure(
            data=[
                go.Bar(
                    x=sorted_df['Team'], 
                    y=sorted_df[metric],
                    orientation='v'
                )
            ]
        )
        
        # Update layout for the chart
        fig.update_layout(
            title=f"{metric} by Team (Top to Bottom)",
            xaxis_title="Team",
            yaxis_title=metric,
            xaxis=dict(categoryorder="total descending")  # Ensure x-axis is sorted from top to bottom
        )
        
        # Display the plot in Streamlit
        st.plotly_chart(fig)

def vocabulary():
    st.markdown("""
    ## 🟡 **How to Understand This Dashboard**

    ### ⚽ **What You’ll Find Here**

    1. **Game State Analysis**
       - Understand how much time the team spends in different match situations:
         - Horsens (**Horsens ahead**)
         - Draw (**equal score**)
         - Opponent (**Opponent ahead**)
       - This helps identify how well the team manages various match scenarios.

    2. **Key Performance Metrics**
       - **xG (Expected Goals):** Measures the quality of chances created. A higher xG means the team is creating better chances.
       - **xG Difference:** The balance between chances created and conceded.
       - **xG Against:** Expected goals conceded — useful for evaluating defensive strength.
       - **Passes per Possession:** Shows how many passes a team has per possession. Between 5 and 9 is the optimal for creating chances.

    3. **Per 90 Minutes Metrics**
       - Standardizes the performance numbers to a per-game basis (per 90 minutes), making it easier to compare across matches or teams.

    ---


    ### 🟠 **Team Mentality Score**
       - Reflects how well the team controls space when defending.
       - Focuses on **limiting dangerous actions** from the opponent in advanced areas.
       - A higher score means better defensive discipline and mentality.
       - We want to avoid box entries and dangerous shots. The team mentality score is measuring how many actions we can handle on our own third without the opponents threatening our goal

    ### 🟢 **Defensive Line Success Rate**
       - Measures how effective the team is at holding a high defensive line.
       - Indicates organization and the ability to limit opponent progress through compactness and positioning.
       - We want to push our defensive line up to 25 meters in front of the goal whenever the ball is more than 40 meters away from our goal
    ### 🔵 **Set-Piece Efficiency**
       - Provides xG and actual goals from set-pieces like:
         - Corners
         - Free kicks
         - Throw-ins
        The definition of a set piece is 10 actions after the set piece or if the ball is cleared away from the final third
    ---"""
)

def player_profiles():
    st.header("Player Profiles & Rating System")

    st.markdown("""
### 🎯 **Player Profiles Explained**

This dashboard breaks down players by position and role using in-game statistics. Players are evaluated on key tactical and technical responsibilities, and **rated on a 1–10 scale using percentile-based scoring**.

---

### 🧮 **How Are Players Rated?**

- Metrics are ranked across all players at the same position.
- **Top 10% = Score of 10**, **Bottom 10% = Score of 1**.
- Some metrics (like possession loss, opponent xG) are inverted → **lower values give higher scores**.
- Each metric belongs to a **category** (Defending, Passing, Chance Creation, Goalscoring, Possession Value).
- All player ratings are calculated based on all games this season, in the same league by players playing the same position.
---

### ⚖️ **How Is the Total Score Calculated?**

- Category scores are combined using **weighted averages**.
- **For defenders and midfielders:**  
  → **Lower-performing areas** are weighted higher (to highlight weaknesses).  
- **For attackers:**  
  → **Top-performing areas** are weighted higher (to emphasize strengths).

---

""")

    with st.expander("Central Defender"):
        st.markdown("""
- **Focus:** Defensive awareness + build-up reliability  
- **Rated on:**  
  - Opponent suppression (xG, xA, PV)
  - Defensive actions: duels won %, aerial duels, interceptions, ball recoveries
  - Possession value added per 90
  - Forward zone pass %, passing % (safe/back zone passing)
  - Limiting attempts conceded inside the box
  - Low possession loss per 90

- **Weighting of categories:**
  - 🛡️ **Defending:** 50%
  - 🎯 **Passing:** 30%
  - ⚡ **Possession Value Added:** 20%
""")

    with st.expander("Fullbacks"):
        st.markdown("""
- **Focus:** Supporting both defense and attack from wide areas  
- **Rated on:**  
  - Defensive duels, aerial duels
  - Interceptions and recoveries
  - Opponent suppression (xG, xA, PV)
  - Final third entries, penalty area entries, assists, total crosses
  - xA per 90, progressive passes
  - Possession value added per 90, low possession loss

- **Weighting of categories:**
  - 🛡️ **Defending:** 30%
  - 🎯 **Passing:** 25%
  - 🎨 **Chance Creation:** 30%
  - ⚡ **Possession Value Added:** 15%
""")

    with st.expander("Number 6"):
        st.markdown("""
- **Focus:** Anchoring midfield, controlling tempo  
- **Rated on:**  
  - Duels won %, aerial duels, interceptions, recoveries
  - Passing security (back zone %, total passes)
  - Forward passing (zone entry, progression)
  - Possession value added per 90
  - Low possession loss

- **Weighting of categories:**
  - 🛡️ **Defending:** 40%
  - 🎯 **Passing:** 35%
  - ⚡ **Possession Value Added:** 25%
""")

    with st.expander("Number 8"):
        st.markdown("""
- **Focus:** Two-way midfielder contributing both defensively and offensively  
- **Rated on:**  
  - Defensive work rate (duels, recoveries, interceptions)
  - Progressive ball movement (forward passes, entries)
  - Chance creation (xA, assists, penalty area entries)
  - Passing efficiency (forward %, back %, total volume)
  - Possession value added

- **Weighting of categories:**
  - 🛡️ **Defending:** 30%
  - 🎯 **Passing:** 30%
  - 🎨 **Progressive Ball Movement / Chance Creation:** 30%
  - ⚡ **Possession Value Added:** 10%
""")

    with st.expander("Number 10"):
        st.markdown("""
- **Focus:** Creativity and offensive production  
- **Rated on:**  
  - Chance creation (xA, assists, open play passes, penalty area entries)
  - Goalscoring contribution (xG, post-shot xG, touches in box)
  - Dribbling and passing in the final third
  - Possession value (PV added, total, loss management)

- **Weighting of categories:**
  - 🎨 **Chance Creation:** 40%
  - 🎯 **Passing:** 25%
  - 🎯 **Goalscoring:** 25%
  - ⚡ **Possession Value Added:** 10%
""")

    with st.expander("Winger"):
        st.markdown("""
- **Focus:** Wide attacking threat and 1v1 capability  
- **Rated on:**  
  - Dribbling success %, dribbles per 90
  - Chance creation (crosses, xA, assists, penalty area entries)
  - Passing quality (forward zone %, volume)
  - Goal threat (xG, post-shot xG, touches in the box)
  - Possession value added, minimizing loss

- **Weighting of categories:**
  - 🎨 **Chance Creation:** 35%
  - 🏹 **Goalscoring:** 25%
  - 🎯 **Passing:** 25%
  - ⚡ **Possession Value Added:** 15%
""")

    with st.expander("Striker"):
        st.markdown("""
- **Focus:** Goalscoring + linking play  
- **Rated on:**  
  - Finishing quality (xG, post-shot xG, attempts in box)
  - Link-up play (passing %, forward zone passing, assists)
  - Possession value added

- **Weighting of categories:**
  - 🏹 **Goalscoring:** 50%
  - 🎯 **Link-up Passing / Chance Creation:** 30%
  - ⚡ **Possession Value Added:** 20%
""")

Data_types = {
    'Dashboard': Dashboard,
    'Opposition analysis': Opposition_analysis,
    'Physical data': Physical_data,
    'Vocabulary': vocabulary,
    'Player profiles': player_profiles
}


st.cache_data()
st.cache_resource()
selected_data = st.sidebar.radio('Choose data type',list(Data_types.keys()))

st.cache_data()
st.cache_resource()
Data_types[selected_data]()
