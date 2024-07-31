import streamlit as st
import pandas as pd
from mplsoccer import Pitch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import ast
from collections import Counter
from scipy.ndimage import gaussian_filter
from datetime import datetime

st.set_page_config(layout='wide')

@st.cache_data
def load_packing_data():
    df_packing = pd.read_csv(r'DNK_1_Division_2024_2025/packing_all DNK_1_Division_2024_2025.csv')
    df_packing['label'] = (df_packing['label'] + ' ' + df_packing['date']).astype(str)
    df_packing = df_packing.rename(columns={'teamName': 'team_name'})
    df_packing['pass_receiver'] = df_packing['pass_receiver'].astype(str)
    df_packing = df_packing[df_packing['pass_receiver'] != '']
    df_packing = df_packing[df_packing['pass_receiver'] != None]
    df_packing = df_packing[df_packing['bypassed_opponents'] < 11]

    return df_packing    
@st.cache_data
def load_spacecontrol_data():
    df_spacecontrol = pd.read_csv(r'DNK_1_Division_2024_2025/Space_control_all DNK_1_Division_2024_2025.csv')
    df_spacecontrol['label'] = (df_spacecontrol['label'] + ' ' + df_spacecontrol['date']).astype(str)
    df_spacecontrol = df_spacecontrol.rename(columns={'teamName': 'team_name'})
    return df_spacecontrol   
@st.cache_data
def load_match_stats(columns=None):
    match_stats = pd.read_csv(r'DNK_1_Division_2024_2025/matchstats_all DNK_1_Division_2024_2025.csv')
    match_stats['label'] = (match_stats['label'] + ' ' + match_stats['date'])
    return match_stats
@st.cache_data
def load_possession_data():
    df_possession = pd.read_csv(r'DNK_1_Division_2024_2025/Horsens/Horsens_possession_data.csv')
    df_possession['label'] = (df_possession['label'] + ' ' + df_possession['date']).astype(str)
    #df_possession['team_name'].str.replace(' ', '_')
    return df_possession
@st.cache_data
def load_modstander():
    team_names = ['AaB','B_93','Fredericia','HB_Køge','Helsingør','Hillerød','Hobro','Horsens','Kolding','Næstved','SønderjyskE','Vendsyssel']  # Replace with your list of team names
    Modstander = st.selectbox('Choose opponent',team_names)
    return Modstander
@st.cache_data
def load_possession_stats():
    df_possession_stats = pd.read_csv(r'DNK_1_Division_2024_2025/possession_stats_all DNK_1_Division_2024_2025.csv')
    df_possession_stats['label'] = (df_possession_stats['label'] + ' ' + df_possession_stats['date'])
    return df_possession_stats
@st.cache_data
def load_xg():
    df_xg = pd.read_csv(r'DNK_1_Division_2024_2025/Horsens/Horsens_xg_data.csv')
    df_xg['label'] = (df_xg['label'] + ' ' + df_xg['date'])
    df_xg['team_name'].str.replace(' ', '_')
    df_xg = df_xg[['playerName','label','team_name','x','y','321','periodId','timeMin','timeSec','9','24','25','26']]
    return df_xg
@st.cache_data
def load_all_xg():
    df_xg_all = pd.read_csv(r'DNK_1_Division_2024_2025/xg_all DNK_1_Division_2024_2025.csv')
    df_xg_all['label'] = (df_xg_all['label'] + ' ' + df_xg_all['date'])
    df_xg_all['team_name'].str.replace(' ', '_')
    return df_xg_all
@st.cache_data
def load_pv():
    df_pv = pd.read_csv(r'DNK_1_Division_2024_2025/Horsens/Horsens_pv_data.csv')
    df_pv['label'] = (df_pv['label'] + ' ' + df_pv['date'])
    df_pv['id'] = df_pv['id'].astype(str)
    df_pv['team_name'].str.replace(' ', '_')
    return df_pv
@st.cache_data
def load_xA():
    df_xA = pd.read_csv(f'DNK_1_Division_2024_2025/xA_all DNK_1_Division_2024_2025.csv')
    df_xA['label'] = (df_xA['label'] + ' ' + df_xA['date']).astype(str)
    return df_xA
@st.cache_data
def counterpressing():
    df_counterpressing = pd.read_csv(r'DNK_1_Division_2024_2025/Horsens/Horsens_counterpressing.csv')
    df_counterpressing['label'] = (df_counterpressing['label'] + ' ' + df_counterpressing['date']).astype(str)
    return df_counterpressing
@st.cache_data
def load_ppda():
    df_ppda = pd.read_csv(r'DNK_1_Division_2024_2025/ppda_all DNK_1_Division_2024_2025.csv')
    df_ppda['label'] = (df_ppda['label'] + ' ' + df_ppda['date']).astype(str)
    return df_ppda
@st.cache_data
def load_crosses():
    df_crosses = pd.read_csv(r'DNK_1_Division_2024_2025/Horsens/Horsens_crosses.csv')
    df_crosses['label'] = (df_crosses['label'] + ' ' + df_crosses['date']).astype(str)
    return df_crosses
@st.cache_data
def load_pv_all():
    df_pv_all = pd.read_csv(r'DNK_1_Division_2024_2025//pv_all DNK_1_Division_2024_2025.csv')
    df_pv_all['label'] = (df_pv_all['label'] + ' ' + df_pv_all['date']).astype(str)
    return df_pv_all
@st.cache_data
def load_squads():
    squads = pd.read_csv(r'DNK_1_Division_2024_2025/squads DNK_1_Division_2024_2025.csv')
    return squads
@st.cache_data
def load_physical_data():
    physical_data = pd.read_csv(r'DNK_1_Division_2024_2025/physical_summary.csv')
    return physical_data

def Process_data_spillere(df_xA,df_pv_all,df_match_stats,df_xg_all,squads):

    def calculate_score(df, column, score_column):
        df_unique = df.drop_duplicates(column).copy()
        df_unique.loc[:, score_column] = pd.qcut(df_unique[column], q=10, labels=False, duplicates='raise') + 1
        return df.merge(df_unique[[column, score_column]], on=column, how='left')
    
    minutter_kamp = 45
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
    df_matchstats = df_match_stats[['player_matchName','player_playerId','contestantId','duelLost','aerialLost','player_position','player_positionSide','successfulOpenPlayPass','totalContest','duelWon','penAreaEntries','accurateBackZonePass','possWonDef3rd','wonContest','accurateFwdZonePass','openPlayPass','totalBackZonePass','minsPlayed','fwdPass','finalThirdEntries','ballRecovery','totalFwdZonePass','successfulFinalThirdPasses','totalFinalThirdPasses','attAssistOpenplay','aerialWon','totalAttAssist','possWonMid3rd','interception','totalCrossNocorner','interceptionWon','attOpenplay','touchesInOppBox','attemptsIbox','totalThroughBall','possWonAtt3rd','accurateCrossNocorner','bigChanceCreated','accurateThroughBall','totalLayoffs','accurateLayoffs','totalFastbreak','shotFastbreak','formationUsed','label','match_id','date']]
    df_matchstats = df_matchstats.rename(columns={'player_matchName': 'playerName'})
    df_scouting = df_matchstats.merge(df_kamp)
    df_xg = df_xg_all[['contestantId','team_name','playerName','playerId','321','match_id','label','date']]
    df_xg = df_xg.rename(columns={'321': 'xg'})
    df_xg['xg'] = df_xg['xg'].astype(float)
    df_xg = df_xg.groupby(['playerName','playerId','match_id','contestantId','team_name','label','date']).sum()
    df_xg = df_xg.reset_index()
    df_scouting = df_scouting.rename(columns={'player_playerId': 'playerId'})
    df_scouting = df_scouting.merge(df_xg, how='left', on=['playerName', 'playerId', 'match_id', 'contestantId', 'team_name', 'label', 'date']).reset_index()
    df_scouting = df_scouting.merge(df_possession_xa_summed, how='left')
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
    
    df_scouting['xg_per90'] = (df_scouting['xg'].astype(float) / df_scouting['minsPlayed'].astype(float)) * 90
    df_scouting['xA_per90'] = (df_scouting['xA'].astype(float) / df_scouting['minsPlayed'].astype(float)) * 90
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

    #df_scouting.fillna(0, inplace=True)

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
        df_balanced_central_defender.loc[:,'minsPlayed'] = df_balanced_central_defender['minsPlayed'].astype(int)
        df_balanced_central_defender = df_balanced_central_defender[df_balanced_central_defender['minsPlayed'].astype(int) >= minutter_kamp]
        
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'duels won %', 'duels won % score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Duels_per90', 'duelWon score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Ballrecovery_per90', 'ballRecovery score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Aerial duel %', 'Aerial duel % score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'aerialWon_per90', 'Aerial duel score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender,'possessionValue.pvAdded_per90', 'Possession value added score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Passing %', 'Open play passing % score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Passes_per90', 'Passing score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Back zone pass %', 'Back zone pass % score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Back zone pass_per90', 'Back zone pass score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Forward zone pass %', 'Forward zone pass % score')
        df_balanced_central_defender = calculate_score(df_balanced_central_defender, 'Forward zone pass_per90', 'Forward zone pass score')

        df_balanced_central_defender['Defending'] = df_balanced_central_defender[['duels won % score','duels won % score','duelWon score','Aerial duel % score','Aerial duel % score','Aerial duel score', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score', 'ballRecovery score']].mean(axis=1)
        df_balanced_central_defender['Possession value added'] = df_balanced_central_defender['Possession value added score']
        df_balanced_central_defender['Passing'] = df_balanced_central_defender[['Open play passing % score','Passing score', 'Back zone pass % score','Back zone pass score','Forward zone pass % score','Forward zone pass score','Possession value added score','Possession value added score']].mean(axis=1)
        
        df_balanced_central_defender['Total score'] = df_balanced_central_defender[['Defending','Defending','Defending','Possession value added','Possession value added','Passing']].mean(axis=1)

        df_balanced_central_defender = df_balanced_central_defender[['playerName','team_name','player_position','label','minsPlayed','age_today','Defending','Possession value added','Passing','Total score']]
        
        df_balanced_central_defendertotal = df_balanced_central_defender[['playerName','team_name','player_position','minsPlayed','age_today','Defending','Possession value added','Passing','Total score']]
        df_balanced_central_defendertotal = df_balanced_central_defendertotal.groupby(['playerName','team_name','player_position','age_today']).mean().reset_index()
        minutter = df_balanced_central_defender.groupby(['playerName', 'team_name','player_position','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_balanced_central_defendertotal['minsPlayed total'] = minutter['minsPlayed']
        df_balanced_central_defender = df_balanced_central_defender.sort_values('Total score',ascending = False)
        df_balanced_central_defendertotal = df_balanced_central_defendertotal[['playerName','team_name','player_position','age_today','minsPlayed total','Defending','Possession value added','Passing','Total score']]
        df_balanced_central_defendertotal = df_balanced_central_defendertotal[df_balanced_central_defendertotal['minsPlayed total'].astype(int) >= minutter_total]
        df_balanced_central_defendertotal = df_balanced_central_defendertotal.sort_values('Total score',ascending = False)

        return df_balanced_central_defender
    
    def fullbacks():
        df_backs = df_scouting[((df_scouting['player_position'] == 'Defender') | (df_scouting['player_position'] == 'Wing Back')) & ((df_scouting['player_positionSide'] == 'Right') | (df_scouting['player_positionSide'] == 'Left'))]
        df_backs.loc[:,'minsPlayed'] = df_backs['minsPlayed'].astype(int)
        df_backs = df_backs[df_backs['minsPlayed'].astype(int) >= minutter_kamp]

        df_backs = calculate_score(df_backs,'possessionValue.pvAdded_per90', 'Possession value added score')
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
        
        df_backs['Defending'] = df_backs[['duels won % score','Duels per 90 score','Duels per 90 score','duels won % score','possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score']].mean(axis=1)
        df_backs['Passing'] = df_backs[['Forward zone pass % score','Forward zone pass per 90 score','finalThird passes % score','finalThirdEntries_per90 score','Back zone pass % score','Back zone pass_per90 score','Possession value added score']].mean(axis=1)
        df_backs['Chance creation'] = df_backs[['Penalty area entries & crosses & shot assists score','totalCrossNocorner_per90 score','totalCrossNocorner_per90 score','finalThirdEntries_per90 score','finalThirdEntries_per90 score','Forward zone pass % score','Forward zone pass per 90 score','Forward zone pass per 90 score','Forward zone pass % score','Possession value added score','Possession value added score']].mean(axis=1)
        df_backs['Possession value added'] = df_backs['Possession value added score']
        
        df_backs = calculate_score(df_backs, 'Defending', 'Defending_')
        df_backs = calculate_score(df_backs, 'Passing', 'Passing_')
        df_backs = calculate_score(df_backs, 'Chance creation','Chance_creation')
        df_backs = calculate_score(df_backs, 'Possession value added', 'Possession_value_added')
        
        df_backs['Total score'] = df_backs[['Defending_','Defending_','Defending_','Defending_','Passing_','Passing_','Chance_creation','Chance_creation','Chance_creation','Possession_value_added','Possession_value_added','Possession_value_added','Possession_value_added']].mean(axis=1)
        df_backs = df_backs[['playerName','team_name','player_position','player_positionSide','label','minsPlayed','age_today','Defending_','Passing_','Chance_creation','Possession_value_added','Total score']]
        df_backs = df_backs.dropna()
        df_backstotal = df_backs[['playerName','team_name','player_position','player_positionSide','minsPlayed','age_today','Defending_','Passing_','Chance_creation','Possession_value_added','Total score']]
        df_backstotal = df_backstotal.groupby(['playerName','team_name','player_position','player_positionSide','age_today']).mean().reset_index()
        minutter = df_backs.groupby(['playerName', 'team_name','player_position','player_positionSide','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_backstotal['minsPlayed total'] = minutter['minsPlayed']
        df_backs = df_backs.sort_values('Total score',ascending = False)
        df_backstotal = df_backstotal[['playerName','team_name','player_position','player_positionSide','age_today','minsPlayed total','Defending_','Passing_','Chance_creation','Possession_value_added','Total score']]
        df_backstotal = df_backstotal[df_backstotal['minsPlayed total'].astype(int) >= minutter_total]
        df_backstotal = df_backstotal.sort_values('Total score',ascending = False)

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

        
        df_sekser['Defending'] = df_sekser[['duels won % score','Duels per 90 score','possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score','possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score','ballRecovery score']].mean(axis=1)
        df_sekser['Passing'] = df_sekser[['Back zone pass % score','Back zone pass_per90 score','Passing % score','Passing score']].mean(axis=1)
        df_sekser['Progressive ball movement'] = df_sekser[['Possession value added score','Possession value added score','Forward zone pass % score','Forward zone pass_per90 score','finalThirdEntries_per90 score']].mean(axis=1)
        df_sekser['Possession value added'] = df_sekser['Possession value added score']
        
        df_sekser = calculate_score(df_sekser, 'Defending', 'Defending_')
        df_sekser = calculate_score(df_sekser, 'Passing', 'Passing_')
        df_sekser = calculate_score(df_sekser, 'Progressive ball movement','Progressive_ball_movement')
        df_sekser = calculate_score(df_sekser, 'Possession value added', 'Possession_value_added')
        
        df_sekser['Total score'] = df_sekser[['Defending_','Passing_','Progressive_ball_movement','Possession_value_added']].mean(axis=1)
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
        df_otter = df_scouting[(df_scouting['player_position'] == 'Midfielder') & df_scouting['player_positionSide'].str.contains('Centre')]
        df_otter.loc[:,'minsPlayed'] = df_otter['minsPlayed'].astype(int)
        df_otter = df_otter[df_otter['minsPlayed'].astype(int) >= minutter_kamp]

        df_otter = calculate_score(df_otter,'Possession value total per_90','Possession value total score')
        df_otter = calculate_score(df_otter,'possessionValue.pvValue_per90', 'Possession value score')
        df_otter = calculate_score(df_otter,'possessionValue.pvAdded_per90', 'Possession value added score')
        df_otter = calculate_score(df_otter, 'duels won %', 'duels won % score')
        df_otter = calculate_score(df_otter,'Duels_per90', 'Duels per 90 score')
        df_otter = calculate_score(df_otter, 'Passing %', 'Passing % score')
        df_otter = calculate_score(df_otter, 'Passes_per90', 'Passing score')
        df_otter = calculate_score(df_otter, 'Back zone pass %', 'Back zone pass % score')
        df_otter = calculate_score(df_otter, 'Back zone pass_per90', 'Back zone pass score')
        df_otter = calculate_score(df_otter, 'finalThirdEntries_per90', 'finalThirdEntries_per90 score')
        df_otter = calculate_score(df_otter, 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90', 'possWonDef3rd_possWonMid3rd_per90&interceptions_per90 score')
        df_otter = calculate_score(df_otter, 'possWonDef3rd_possWonMid3rd_possWonAtt3rd_per90', 'possWonDef3rd_possWonMid3rd_possWonAtt3rd_per90 score')
        df_otter = calculate_score(df_otter, 'Forward zone pass %', 'Forward zone pass % score')
        df_otter = calculate_score(df_otter, 'Forward zone pass_per90', 'Forward zone pass score')
        df_otter = calculate_score(df_otter, 'fwdPass_per90', 'fwd_Pass_per90 score')
        df_otter = calculate_score(df_otter, 'attAssistOpenplay_per90','attAssistOpenplay_per90 score')
        df_otter = calculate_score(df_otter, 'penAreaEntries_per90','penAreaEntries_per90 score')

        df_otter['Defending'] = df_otter[['duels won % score','Duels per 90 score','possWonDef3rd_possWonMid3rd_possWonAtt3rd_per90 score']].mean(axis=1)
        df_otter['Passing'] = df_otter[['Forward zone pass % score','Forward zone pass score','Passing % score','Passing score']].mean(axis=1)
        df_otter['Progressive ball movement'] = df_otter[['attAssistOpenplay_per90 score','fwd_Pass_per90 score','penAreaEntries_per90 score','Forward zone pass % score','Forward zone pass score','finalThirdEntries_per90 score','Possession value total score']].mean(axis=1)
        df_otter['Possession value'] = df_otter[['Possession value added score','Possession value total score']].mean(axis=1)
        
        df_otter = calculate_score(df_otter, 'Defending', 'Defending_')
        df_otter = calculate_score(df_otter, 'Passing', 'Passing_')
        df_otter = calculate_score(df_otter, 'Progressive ball movement','Progressive_ball_movement')
        df_otter = calculate_score(df_otter, 'Possession value', 'Possession_value')
        
        df_otter['Total score'] = df_otter[['Defending_','Passing_','Passing_','Progressive_ball_movement','Progressive_ball_movement','Progressive_ball_movement','Progressive_ball_movement','Possession_value','Possession_value','Possession_value']].mean(axis=1)
        df_otter = df_otter[['playerName','team_name','player_position','label','minsPlayed','age_today','Defending_','Passing_','Progressive_ball_movement','Possession_value','Total score']]
        df_otter = df_otter.dropna()

        df_ottertotal = df_otter[['playerName','team_name','player_position','minsPlayed','age_today','Defending_','Passing_','Progressive_ball_movement','Possession_value','Total score']]

        df_ottertotal = df_ottertotal.groupby(['playerName','team_name','player_position','age_today']).mean().reset_index()
        minutter = df_otter.groupby(['playerName', 'team_name','player_position','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_ottertotal['minsPlayed total'] = minutter['minsPlayed']
        df_otter = df_otter.sort_values('Total score',ascending = False)
        df_ottertotal = df_ottertotal[['playerName','team_name','player_position','age_today','minsPlayed total','Defending_','Passing_','Progressive_ball_movement','Possession_value','Total score']]
        df_ottertotal= df_ottertotal[df_ottertotal['minsPlayed total'].astype(int) >= minutter_total]
        df_ottertotal = df_ottertotal.sort_values('Total score',ascending = False)

        return df_otter
        
    def number10():
        df_10 = df_scouting[
            ((df_scouting['player_position'] == 'Attacking Midfielder') & df_scouting['player_positionSide'].str.contains('Centre')) |
            ((df_scouting['player_position'] == 'Midfielder') & df_scouting['player_positionSide'].str.contains('Centre'))]
        df_10.loc[:,'minsPlayed'] = df_10['minsPlayed'].astype(int)
        df_10 = df_10[df_10['minsPlayed'].astype(int) >= minutter_kamp]

        df_10 = calculate_score(df_10,'Possession value total per_90','Possession value total score')
        df_10 = calculate_score(df_10,'possessionValue.pvValue_per90', 'Possession value score')
        df_10 = calculate_score(df_10,'possessionValue.pvAdded_per90', 'Possession value added score')
        df_10 = calculate_score(df_10, 'Passing %', 'Passing % score')
        df_10 = calculate_score(df_10, 'Passes_per90', 'Passing score')
        df_10 = calculate_score(df_10, 'finalThirdEntries_per90', 'finalThirdEntries_per90 score')
        df_10 = calculate_score(df_10, 'Forward zone pass %', 'Forward zone pass % score')
        df_10 = calculate_score(df_10, 'Forward zone pass_per90', 'Forward zone pass score')
        df_10 = calculate_score(df_10, 'fwdPass_per90', 'fwd_Pass_per90 score')
        df_10 = calculate_score(df_10, 'attAssistOpenplay_per90','attAssistOpenplay_per90 score')
        df_10 = calculate_score(df_10, 'penAreaEntries_per90','penAreaEntries_per90 score')
        df_10 = calculate_score(df_10, 'finalThird passes %','finalThird passes % score')
        df_10 = calculate_score(df_10, 'finalthirdpass_per90', 'finalthirdpass per 90 score')
        df_10 = calculate_score(df_10, 'dribble %','dribble % score')
        df_10 = calculate_score(df_10, 'dribble_per90','dribble score')
        df_10 = calculate_score(df_10, 'touches_in_box_per90','touches_in_box_per90 score')
        df_10 = calculate_score(df_10, 'xA_per90','xA_per90 score')
        df_10 = calculate_score(df_10, 'xg_per90','xg_per90 score')


        df_10['Passing'] = df_10[['Forward zone pass % score','Forward zone pass score','Passing % score','Passing score']].mean(axis=1)
        df_10['Chance creation'] = df_10[['attAssistOpenplay_per90 score','penAreaEntries_per90 score','Forward zone pass % score','Forward zone pass score','finalThird passes % score','finalthirdpass per 90 score','Possession value total score','Possession value score','dribble % score','touches_in_box_per90 score','xA_per90 score']].mean(axis=1)
        df_10['Goalscoring'] = df_10[['xg_per90 score','xg_per90 score','touches_in_box_per90 score']].mean(axis=1)
        df_10['Possession value'] = df_10[['Possession value total score','Possession value total score','Possession value added score','Possession value score']].mean(axis=1)
                
        df_10 = calculate_score(df_10, 'Passing', 'Passing_')
        df_10 = calculate_score(df_10, 'Chance creation','Chance_creation')
        df_10 = calculate_score(df_10, 'Goalscoring','Goalscoring_')        
        df_10 = calculate_score(df_10, 'Possession value', 'Possession_value')
        
        df_10['Total score'] = df_10[['Passing_','Chance_creation','Chance_creation','Chance_creation','Chance_creation','Goalscoring_','Goalscoring_','Goalscoring_','Possession_value','Possession_value','Possession_value']].mean(axis=1)
        df_10 = df_10[['playerName','team_name','label','minsPlayed','age_today','Passing_','Chance_creation','Goalscoring_','Possession_value','Total score']]
        df_10 = df_10.dropna()
        df_10total = df_10[['playerName','team_name','minsPlayed','age_today','Passing_','Chance_creation','Goalscoring_','Possession_value','Total score']]

        df_10total = df_10total.groupby(['playerName','team_name','age_today']).mean().reset_index()
        minutter = df_10.groupby(['playerName', 'team_name','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_10total['minsPlayed total'] = minutter['minsPlayed']
        df_10 = df_10.sort_values('Total score',ascending = False)
        df_10total = df_10total[['playerName','team_name','age_today','minsPlayed total','Passing_','Chance_creation','Goalscoring_','Possession_value','Total score']]
        df_10total= df_10total[df_10total['minsPlayed total'].astype(int) >= minutter_total]
        df_10total = df_10total.sort_values('Total score',ascending = False)

        return df_10
    
    def winger():
        df_10 = df_scouting[
            (
                (df_scouting['player_position'] == 'Midfielder') & 
                (df_scouting['player_positionSide'].isin(['Right', 'Left']))
            ) |
            (
                (df_scouting['player_position'].isin(['Attacking Midfielder', 'Striker'])) &
                (df_scouting['player_positionSide'].isin(['Right', 'Left']))
            )
        ]        
        df_10.loc[:,'minsPlayed'] = df_10['minsPlayed'].astype(int)
        df_10 = df_10[df_10['minsPlayed'].astype(int) >= minutter_kamp]

        df_10 = calculate_score(df_10,'Possession value total per_90','Possession value total score')
        df_10 = calculate_score(df_10,'possessionValue.pvValue_per90', 'Possession value score')
        df_10 = calculate_score(df_10,'possessionValue.pvAdded_per90', 'Possession value added score')
        df_10 = calculate_score(df_10, 'Passing %', 'Passing % score')
        df_10 = calculate_score(df_10, 'Passes_per90', 'Passing score')
        df_10 = calculate_score(df_10, 'finalThirdEntries_per90', 'finalThirdEntries_per90 score')
        df_10 = calculate_score(df_10, 'Forward zone pass %', 'Forward zone pass % score')
        df_10 = calculate_score(df_10, 'Forward zone pass_per90', 'Forward zone pass score')
        df_10 = calculate_score(df_10, 'fwdPass_per90', 'fwd_Pass_per90 score')
        df_10 = calculate_score(df_10, 'attAssistOpenplay_per90','attAssistOpenplay_per90 score')
        df_10 = calculate_score(df_10, 'penAreaEntries_per90','penAreaEntries_per90 score')
        df_10 = calculate_score(df_10, 'finalThird passes %','finalThird passes % score')
        df_10 = calculate_score(df_10, 'finalthirdpass_per90', 'finalthirdpass per 90 score')
        df_10 = calculate_score(df_10, 'dribble %','dribble % score')
        df_10 = calculate_score(df_10, 'dribble_per90', 'dribble score')
        df_10 = calculate_score(df_10, 'touches_in_box_per90','touches_in_box_per90 score')
        df_10 = calculate_score(df_10, 'xA_per90','xA_per90 score')
        df_10 = calculate_score(df_10, 'attemptsIbox_per90','attemptsIbox_per90 score')
        df_10 = calculate_score(df_10, 'xg_per90','xg_per90 score')


        df_10['Passing'] = df_10[['Forward zone pass % score','Forward zone pass score','Passing % score','Passing score']].mean(axis=1)
        df_10['Chance creation'] = df_10[['attAssistOpenplay_per90 score','penAreaEntries_per90 score','Forward zone pass % score','Forward zone pass score','finalThird passes % score','finalthirdpass per 90 score','Possession value total score','Possession value score','dribble % score','dribble score','touches_in_box_per90 score','xA_per90 score']].mean(axis=1)
        df_10['Goalscoring'] = df_10[['xg_per90 score','xg_per90 score','touches_in_box_per90 score']].mean(axis=1)
        df_10['Possession value'] = df_10[['Possession value total score','Possession value total score','Possession value added score','Possession value score','Possession value score','Possession value score']].mean(axis=1)
                
        df_10 = calculate_score(df_10, 'Passing', 'Passing_')
        df_10 = calculate_score(df_10, 'Chance creation','Chance_creation')
        df_10 = calculate_score(df_10, 'Goalscoring','Goalscoring_')        
        df_10 = calculate_score(df_10, 'Possession value', 'Possession_value')
        
        df_10['Total score'] = df_10[['Passing_','Chance_creation','Chance_creation','Chance_creation','Chance_creation','Goalscoring_','Goalscoring_','Goalscoring_','Goalscoring_','Possession_value','Possession_value','Possession_value','Possession_value']].mean(axis=1)
        df_10 = df_10[['playerName','team_name','label','minsPlayed','age_today','Passing_','Chance_creation','Goalscoring_','Possession_value','Total score']]
        df_10 = df_10.dropna()
        df_10total = df_10[['playerName','team_name','minsPlayed','age_today','Passing_','Chance_creation','Goalscoring_','Possession_value','Total score']]

        df_10total = df_10total.groupby(['playerName','team_name','age_today']).mean().reset_index()
        minutter = df_10.groupby(['playerName', 'team_name','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_10total['minsPlayed total'] = minutter['minsPlayed']
        df_kant = df_10.sort_values('Total score',ascending = False)
        df_10total = df_10total[['playerName','team_name','age_today','minsPlayed total','Passing_','Chance_creation','Goalscoring_','Possession_value','Total score']]
        df_10total= df_10total[df_10total['minsPlayed total'].astype(int) >= minutter_total]
        df_10total = df_10total.sort_values('Total score',ascending = False)

        return df_kant
    
    def Classic_striker():
        df_striker = df_scouting[(df_scouting['player_position'] == 'Striker') & (df_scouting['player_positionSide'].str.contains('Centre'))]
        df_striker.loc[:,'minsPlayed'] = df_striker['minsPlayed'].astype(int)
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
        df_striker = calculate_score(df_striker, 'bigChanceCreated_per90','bigChanceCreated_per90 score')
        df_striker = calculate_score(df_striker, 'dribble %','dribble % score')
        df_striker = calculate_score(df_striker, 'dribble_per90', 'dribble_per90 score')
        df_striker = calculate_score(df_striker, 'touches_in_box_per90','touches_in_box_per90 score')
        df_striker = calculate_score(df_striker, 'xA_per90','xA_per90 score')
        df_striker = calculate_score(df_striker, 'attemptsIbox_per90','attemptsIbox_per90 score')
        df_striker = calculate_score(df_striker, 'xg_per90','xg_per90 score')


        df_striker['Linkup_play'] = df_striker[['Forward zone pass % score','Forward zone pass score','Passing % score','Passing score','Possession value score','penAreaEntries_per90 score','finalThirdEntries_per90 score']].mean(axis=1)
        df_striker['Chance_creation'] = df_striker[['penAreaEntries_per90 score','Possession value total score','bigChanceCreated_per90 score','touches_in_box_per90 score','finalThirdEntries_per90 score']].mean(axis=1)
        df_striker['Goalscoring_'] = df_striker[['attemptsIbox_per90 score','xg_per90 score','xg_per90 score','xg_per90 score','xg_per90 score']].mean(axis=1)
        df_striker['Possession_value'] = df_striker[['Possession value total score','Possession value score','Possession value score','Possession value score']].mean(axis=1)

        df_striker = calculate_score(df_striker, 'Linkup_play', 'Linkup play')
        df_striker = calculate_score(df_striker, 'Chance_creation','Chance creation')
        df_striker = calculate_score(df_striker, 'Goalscoring_','Goalscoring')        
        df_striker = calculate_score(df_striker, 'Possession_value', 'Possession value')

        
        df_striker['Total score'] = df_striker[['Linkup play','Chance creation','Goalscoring','Goalscoring','Goalscoring','Possession value']].mean(axis=1)
        df_striker = df_striker[['playerName','team_name','label','minsPlayed','age_today','Linkup play','Chance creation','Goalscoring','Possession value','Total score']]
        df_striker = df_striker.dropna()

        df_strikertotal = df_striker[['playerName','team_name','minsPlayed','age_today','Linkup play','Chance creation','Goalscoring','Possession value','Total score']]

        df_strikertotal = df_strikertotal.groupby(['playerName','team_name','age_today']).mean().reset_index()
        minutter = df_striker.groupby(['playerName', 'team_name','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_strikertotal['minsPlayed total'] = minutter['minsPlayed']
        df_classic_striker = df_striker.sort_values('Total score',ascending = False)
        df_strikertotal = df_strikertotal[['playerName','team_name','age_today','minsPlayed total','Linkup play','Chance creation','Goalscoring','Possession value','Total score']]
        df_strikertotal= df_strikertotal[df_strikertotal['minsPlayed total'].astype(int) >= minutter_total]
        df_strikertotal = df_strikertotal.sort_values('Total score',ascending = False)
        return df_classic_striker
    
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

df_xA = load_xA()
df_pv = load_pv_all()
df_match_stats = load_match_stats()
df_xg_all = load_all_xg()
squads = load_squads()

position_dataframes = Process_data_spillere(df_xA, df_pv, df_match_stats, df_xg_all, squads)
balanced_central_defender_df = position_dataframes['Central defender']
fullbacks_df = position_dataframes['Fullbacks']
number6_df = position_dataframes['Number 6']
number8_df = position_dataframes['Number 8']
number10_df = position_dataframes['Number 10']
winger_df = position_dataframes['Winger']
classic_striker_df = position_dataframes['Classic striker']

def Dashboard():
    df_xg = load_xg()
    df_pv = load_pv()
    df_possession_stats = load_possession_stats()
    df_possession = load_possession_data()
    df_possession['team_name'] = df_possession['team_name'].apply(lambda x: x if x == 'Horsens' else 'Opponent')
    df_xg['team_name'] = df_xg['team_name'].apply(lambda x: x if x == 'Horsens' else 'Opponent')

    df_matchstats = load_match_stats()
    df_packing = load_packing_data()
    df_xA = load_xA()
    df_spacecontrol = load_spacecontrol_data()
    
    st.title('AC Horsens First Team Dashboard')
    df_possession['date'] = pd.to_datetime(df_possession['date'])
    df_possession = df_possession.sort_values(by='date')
    matches = df_possession['label'].unique()
    matches = matches[::-1]
    match_choice = st.multiselect('Choose a match', matches)
    df_xg = df_xg[df_xg['label'].isin(match_choice)]
    df_pv = df_pv[df_pv['label'].isin(match_choice)]
    df_possession_stats = df_possession_stats[df_possession_stats['label'].isin(match_choice)]
    df_packing = df_packing[df_packing['label'].isin(match_choice)]
    df_matchstats = df_matchstats[df_matchstats['label'].isin(match_choice)]
    df_possession = df_possession[df_possession['label'].isin(match_choice)]
    df_xA = df_xA[df_xA['label'].isin(match_choice)]
    df_spacecontrol = df_spacecontrol[df_spacecontrol['label'].isin(match_choice)]
    df_spacecontrol = df_spacecontrol[df_spacecontrol['Type'] == 'Player']
    df_spacecontrol = df_spacecontrol[['Team','TotalControlArea','CenterControlArea','PenaltyAreaControl','label']]
    df_spacecontrol[['TotalControlArea', 'CenterControlArea', 'PenaltyAreaControl']] = df_spacecontrol[['TotalControlArea', 'CenterControlArea', 'PenaltyAreaControl']].astype(float).round(2)
    df_spacecontrol['Team'] = df_spacecontrol['Team'].apply(lambda x: x if x == 'Horsens' else 'Opponent')
    
    df_spacecontrol = df_spacecontrol.groupby(['Team', 'label']).sum().reset_index()    
    df_spacecontrol['TotalControlArea_match'] = df_spacecontrol.groupby('label')['TotalControlArea'].transform('sum')
    df_spacecontrol['CenterControlArea_match'] = df_spacecontrol.groupby('label')['CenterControlArea'].transform('sum')
    df_spacecontrol['PenaltyAreaControl_match'] = df_spacecontrol.groupby('label')['PenaltyAreaControl'].transform('sum')

    df_spacecontrol['Total Control Area %'] = df_spacecontrol['TotalControlArea'] / df_spacecontrol['TotalControlArea_match'] * 100
    df_spacecontrol['Center Control Area %'] = df_spacecontrol['CenterControlArea'] / df_spacecontrol['CenterControlArea_match'] * 100
    df_spacecontrol['Penalty Area Control %'] = df_spacecontrol['PenaltyAreaControl'] / df_spacecontrol['PenaltyAreaControl_match'] * 100
    df_spacecontrol = df_spacecontrol[['Team', 'label', 'Total Control Area %', 'Center Control Area %', 'Penalty Area Control %']]
    df_spacecontrol = df_spacecontrol.rename(columns={'Team': 'team_name'})

    xA_map = df_xA[['contestantId','team_name']]
    df_matchstats = df_matchstats.merge(xA_map, on='contestantId', how='inner')
    df_matchstats = df_matchstats.drop_duplicates()
    df_matchstats['team_name'] = df_matchstats['team_name'].apply(lambda x: x if x == 'Horsens' else 'Opponent')
    df_passes = df_matchstats[['team_name','label','openPlayPass','successfulOpenPlayPass']]

    df_passes = df_passes.groupby(['team_name','label']).sum().reset_index()

    df_xA_summary = df_possession.groupby(['team_name','label'])['318.0'].sum().reset_index()
    df_xA_summary = df_xA_summary.rename(columns={'318.0': 'xA'})

    df_xg_summary = df_xg.groupby(['team_name','label'])['321'].sum().reset_index()
    df_xg_summary = df_xg_summary.rename(columns={'321': 'xG'})
    df_packing_summary = df_packing[['team_name','label','bypassed_opponents','bypassed_defenders']]
    df_packing_summary['team_name'] = df_packing_summary['team_name'].apply(lambda x: x if x == 'Horsens' else 'Opponent')

    df_packing_summary = df_packing_summary.groupby(['team_name','label']).sum().reset_index()
    
    team_summary = df_xg_summary.merge(df_xA_summary, on=['team_name','label'])
    team_summary = team_summary.merge(df_passes, on=['team_name','label'])
    team_summary = team_summary.merge(df_packing_summary, on=['team_name', 'label'])
    team_summary = team_summary.merge(df_spacecontrol, on=['team_name', 'label'])
    team_summary = team_summary.drop(columns='label')
    team_summary = team_summary.groupby('team_name').mean().reset_index()
    team_summary = team_summary.round(2)
    st.dataframe(team_summary.style.format(precision=2), use_container_width=True,hide_index=True)

    
    def xg():
        df_xg = load_xg()
        xg_all = load_all_xg()
        xg_all = xg_all[~(xg_all[['9','24', '25', '26']] == True).any(axis=1)]
        
        df_xg = df_xg[['playerName', 'label', 'team_name', 'x', 'y', '321', 'periodId', 'timeMin', 'timeSec', '9', '24', '25', '26']]
        df_xg = df_xg[df_xg['label'].isin(match_choice)]
        df_xg = df_xg[~(df_xg[['9','24', '25', '26']] == True).any(axis=1)]

        xg_period = df_xg[['team_name','321','label']]
        xg_period = xg_period.groupby(['team_name', 'label']).sum().reset_index()
        xg_period['xG_match'] = xg_period.groupby('label')['321'].transform('sum')
        xg_period['xG difference period'] = xg_period['321'] - xg_period['xG_match'] + xg_period['321']
        xg_period = xg_period.groupby('team_name').sum().reset_index()
        xg_period = xg_period[['team_name', 'xG difference period']]
        xg_period = xg_period.sort_values(by=['xG difference period'], ascending=False)
        xg_period = xg_period[xg_period['team_name'] == 'Horsens']
        xg_period['xG difference period'] = xg_period['xG difference period'].round(2)
        
        xg_all = xg_all[['team_name','321','label','date']]
        xg_all = xg_all.groupby(['team_name','label','date']).sum().reset_index()
        xg_all['xG_match'] = xg_all.groupby('label')['321'].transform('sum')
        xg_all['xG difference'] = xg_all['321'] - xg_all['xG_match'] + xg_all['321']
        xg_all = xg_all.sort_values(by=['date'], ascending=True)
        xg_all_table = xg_all.groupby('team_name').sum().reset_index()
        xg_all_table = xg_all_table[['team_name', 'xG difference']]
        xg_all_table = xg_all_table.sort_values(by=['xG difference'], ascending=False)
        xg_all_table['xG difference'] = xg_all_table['xG difference'].round(2)
        xg_all_table['xG difference rank'] = xg_all_table['xG difference'].rank(ascending=False)
        st.header('Whole season')
        st.dataframe(xg_all_table,hide_index=True)

        xg_all['xG rolling average'] = xg_all.groupby('team_name')['xG difference'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        fig = go.Figure()
        
        for team in xg_all['team_name'].unique():
            team_data = xg_all[xg_all['team_name'] == team]
            line_size = 5 if team == 'Horsens' else 1  # Larger line for Horsens
            fig.add_trace(go.Scatter(
                x=team_data['date'], 
                y=team_data['xG rolling average'], 
                mode='lines',
                name=team,
                line=dict(width=line_size)
            ))
        
        fig.update_layout(
            title='3-Game Rolling Average of xG Difference Over Time',
            xaxis_title='Date',
            yaxis_title='3-Game Rolling Average xG Difference',
            template='plotly_white'
        )
        
        st.plotly_chart(fig)

        df_xg['team_name'] = df_xg['team_name'].apply(lambda x: x if x == 'Horsens' else 'Opponent')
        df_xg = df_xg.sort_values(by=['team_name','timeMin'])

        df_xg['cumulative_xG'] = df_xg.groupby(['team_name'])['321'].cumsum()

        fig = go.Figure()
        
        for team in df_xg['team_name'].unique():
            team_data = df_xg[df_xg['team_name'] == team]
            fig.add_trace(go.Scatter(
                x=team_data['timeMin'], 
                y=team_data['cumulative_xG'], 
                mode='lines',
                name=team,
            ))
        
        fig.update_layout(
            title='Average Cumulative xG Over Time',
            xaxis_title='Time (Minutes)',
            yaxis_title='Average Cumulative xG',
            template='plotly_white'
        )
        st.header('Chosen matches')
        st.dataframe(xg_period, hide_index=True)        
        st.plotly_chart(fig)
    
        df_xg_plot = df_xg[['playerName','team_name','x','y', '321']]
        df_xg_plot = df_xg_plot[df_xg_plot['team_name'] == 'Horsens']
        pitch = Pitch(pitch_type='wyscout',half=True,line_color='white', pitch_color='grass')
        fig, ax = pitch.draw(figsize=(10, 6))
        
        sc = ax.scatter(df_xg_plot['x'], df_xg_plot['y'], s=df_xg_plot['321'] * 100, c='red', edgecolors='black', alpha=0.6)
        
        for i, row in df_xg_plot.iterrows():
            ax.text(row['x'], row['y'], f"{row['playerName']}\n{row['321']:.2f}", fontsize=6, ha='center', va='center')
        
        st.pyplot(fig)
        
    def passes():
        
        df_matchstats = load_match_stats(columns=['contestantId','date', 'label', 'successfulOpenPlayPass', 'openPlayPass'])
        df_matchstats['date'] = pd.to_datetime(df_matchstats['date'])
        df_possession = load_possession_data()
        df_xA = load_xA()
        xA_map = df_xA[['contestantId', 'team_name']].drop_duplicates()
        df_matchstats = df_matchstats.merge(xA_map, on='contestantId')
        df_matchstats = df_matchstats[['label','date', 'team_name', 'successfulOpenPlayPass', 'openPlayPass']]
        df_matchstats_tabel = df_matchstats[['team_name', 'successfulOpenPlayPass', 'openPlayPass']]
        df_matchstats_tabel = df_matchstats_tabel.groupby('team_name').sum().reset_index()
        df_matchstats_tabel = df_matchstats_tabel.sort_values(by='openPlayPass', ascending=False)
        df_matchstats = df_matchstats.groupby(['label','date', 'team_name']).sum().reset_index()
        df_matchstats = df_matchstats.sort_values(by='date')
        st.header('Whole season')
        st.dataframe(df_matchstats_tabel, hide_index=True)
        # Beregn 3-kamps rullende gennemsnit for hver team
        
        df_matchstats['rolling_openPlayPass'] = df_matchstats.groupby('team_name')['openPlayPass'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df_matchstats['rolling_successfulOpenPlayPass'] = df_matchstats.groupby('team_name')['successfulOpenPlayPass'].transform(lambda x: x.rolling(3, min_periods=1).mean())

        fig1 = go.Figure()

        for team in df_matchstats['team_name'].unique():
            team_data = df_matchstats[df_matchstats['team_name'] == team]
            line_size = 5 if team == 'Horsens' else 1  # Larger line for Horsens
            fig1.add_trace(go.Scatter(
                x=team_data['date'],
                y=team_data['rolling_openPlayPass'],
                mode='lines',
                name=team,
                line=dict(width=line_size)
            ))

        fig1.update_layout(
            title='3-Game Rolling Average of Open Play Passes',
            xaxis_title='Date',
            yaxis_title='3-Game Rolling Average Open Play Passes',
            template='plotly_white'
        )

        # Plot for successfulOpenPlayPass med rullende gennemsnit
        fig2 = go.Figure()

        for team in df_matchstats['team_name'].unique():
            team_data = df_matchstats[df_matchstats['team_name'] == team]
            line_size = 5 if team == 'Horsens' else 1  # Larger line for Horsens
            fig2.add_trace(go.Scatter(
                x=team_data['date'],
                y=team_data['rolling_successfulOpenPlayPass'],
                mode='lines',
                name=team,
                line=dict(width=line_size)
            ))

        fig2.update_layout(
            title='3-Game Rolling Average of Successful Open Play Passes',
            xaxis_title='Date',
            yaxis_title='3-Game Rolling Average Successful Open Play Passes',
            template='plotly_white'
        )

        # Vis plots i Streamlit
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

        df_possession = df_possession[~(df_possession[['6.0','107.0']] == True).any(axis=1)]
        df_possession = df_possession[df_possession['label'].isin(match_choice)]
        df_passes_horsens = df_possession[df_possession['team_name'] == 'Horsens']
        
        df_passes_horsens = df_passes_horsens.sort_values(by='eventId').reset_index(drop=True)
        df_passes_horsens['pass_receiver'] = None
    
        for i in range(len(df_passes_horsens) - 1):
            current_event = df_passes_horsens.loc[i]
            if current_event['typeId'] == 1 and current_event['outcome'] == 1:
                next_event_id = current_event['eventId'] + 1
                next_event = df_passes_horsens[(df_passes_horsens['eventId'] == next_event_id) & (df_passes_horsens['team_name'] == current_event['team_name'])]

                if not next_event.empty:
                    pass_receiver = next_event.iloc[0]['playerName']
                    df_passes_horsens.at[i, 'pass_receiver'] = pass_receiver
        df_passes_horsens = df_passes_horsens[(df_passes_horsens['typeId'] == 1) & (df_passes_horsens['outcome'] == 1)]

        mid_third_pass_ends = df_passes_horsens[
            (df_passes_horsens['140.0'].astype(float) >= 33.3) & 
            (df_passes_horsens['140.0'].astype(float) <= 66.3) & 
            (df_passes_horsens['141.0'].astype(float) >= 21.1) & 
            (df_passes_horsens['141.0'].astype(float) <= 78.9) & 
            ((df_passes_horsens['y'].astype(float) <= 21.1) | 
            (df_passes_horsens['y'].astype(float) >= 78.9))
        ]
        mid_third_pass_ends = mid_third_pass_ends[['typeId','team_name','playerName','pass_receiver','eventId', '140.0', '141.0','x', 'y','label','date','outcome']]
        
        # Tæl forekomster af kombinationer af team_name og label
        team_counts = mid_third_pass_ends.groupby(['team_name','label']).size().reset_index(name='count')
        team_counts.columns = ['team_name', 'label', 'count']
        team_counts = team_counts.sort_values(by=['count'], ascending=False)

        # Tæl forekomster af hver playerName
        player_counts = mid_third_pass_ends['playerName'].value_counts().reset_index(name='Passed')
        player_counts.columns = ['playerName', 'Passed']
        pass_receiver_counts = mid_third_pass_ends['pass_receiver'].value_counts().reset_index(name='Received')
        pass_receiver_counts.columns = ['pass_receiver', 'Received']
        pass_receiver_counts.rename(columns={'pass_receiver': 'playerName'}, inplace=True)
        player_counts = player_counts.merge(pass_receiver_counts, on='playerName', how='outer')
        player_counts = player_counts.fillna(0)
        player_counts['Total'] = player_counts['Passed'] + player_counts['Received']
        player_counts = player_counts.sort_values(by=['Total'], ascending=False)
        st.header('Chosen matches')
        st.write('Passes from side to halfspace/centerspace')
        st.dataframe(player_counts,hide_index=True)
        st.dataframe(team_counts,hide_index=True)
        option2 = st.selectbox(
            'Select the position',
            ('Start', 'End')
        )

        # Initialize the pitch
        pitch = Pitch(pitch_type='opta',line_zorder=2, pitch_color='grass', line_color='white')
        fig, ax = pitch.draw()

        # Extract coordinates based on user selection
        if option2 == 'Start':
            x_coords = mid_third_pass_ends['x']
            y_coords = mid_third_pass_ends['y']
        else:
            x_coords = mid_third_pass_ends['140.0']
            y_coords = mid_third_pass_ends['141.0']

        # Plot the heatmap
        fig.set_facecolor('#22312b')
        bin_statistic = pitch.bin_statistic(x_coords, y_coords, statistic='count', bins=(50, 50)) # Adjust bins as needed
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')

        pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='black')

        # Display the plot in Streamlit
        st.pyplot(fig)

        
        st.write('Passes from center to side/halfspace on last third')
        final_third_pass_ends = df_passes_horsens[
            (
                (df_passes_horsens['140.0'].astype(float) >= 66.3) & 
                (
                    (df_passes_horsens['141.0'].astype(float) <= 21.1) | 
                    (df_passes_horsens['141.0'].astype(float) >= 78.9)
                )
            ) & 
            (
                ((df_passes_horsens['140.0'].astype(float) >= 66.3) & 
                (df_passes_horsens['y'].astype(float) >= 36.8) & 
                (df_passes_horsens['y'].astype(float) <= 63.2))
            )
        ]
        final_third_pass_ends = final_third_pass_ends[['typeId','team_name','playerName','pass_receiver','eventId', '140.0', '141.0','x', 'y','label','date','outcome']]
        
        # Tæl forekomster af kombinationer af team_name og label
        team_counts = final_third_pass_ends.groupby(['team_name','label']).size().reset_index(name='count')
        team_counts.columns = ['team_name', 'label', 'count']
        team_counts = team_counts.sort_values(by=['count'], ascending=False)

        # Tæl forekomster af hver playerName
        player_counts = final_third_pass_ends['playerName'].value_counts().reset_index(name='Passed')
        player_counts.columns = ['playerName', 'Passed']
        pass_receiver_counts = final_third_pass_ends['pass_receiver'].value_counts().reset_index(name='Received')
        pass_receiver_counts.columns = ['pass_receiver', 'Received']
        pass_receiver_counts.rename(columns={'pass_receiver': 'playerName'}, inplace=True)
        player_counts = player_counts.merge(pass_receiver_counts, on='playerName', how='outer')
        player_counts = player_counts.fillna(0)
        player_counts['Total'] = player_counts['Passed'] + player_counts['Received']
        player_counts = player_counts.sort_values(by=['Total'], ascending=False)
        st.dataframe(player_counts,hide_index=True)
        st.dataframe(team_counts,hide_index=True)
        pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white')
        fig, ax = pitch.draw()

        # Plotting the arrows
        for index, row in final_third_pass_ends.iterrows():
            pitch.arrows(row['x'], row['y'], row['140.0'], row['141.0'], ax=ax, width=2, headwidth=3, color='black')

        st.pyplot(fig)

    def packing():
        df_packing = load_packing_data()
        df_packing['pass_receiver'] = df_packing['pass_receiver'].astype(str)
        df_packing = df_packing[df_packing['pass_receiver'] != '']
        df_packing = df_packing[df_packing['pass_receiver'] != None]
        df_packing = df_packing[df_packing['bypassed_opponents'] < 11]

        packing_teams = df_packing.groupby('team_name')[['bypassed_opponents','bypassed_defenders']].sum().reset_index()
        packing_teams = packing_teams.sort_values(by='bypassed_opponents', ascending=False)
        st.header('Whole season')
        st.dataframe(packing_teams, hide_index=True)
        df_packing_time = df_packing.groupby(['label','date', 'team_name'])['bypassed_opponents'].sum().reset_index()
        df_packing_time = df_packing_time.sort_values(by='date')
        df_packing_time['packing_match'] = df_packing_time.groupby('label')['bypassed_opponents'].transform('sum')
        df_packing_time['packing_diff'] = df_packing_time['bypassed_opponents'] - df_packing_time['packing_match'] + df_packing_time['bypassed_opponents']
        # Beregn 3-kamps rullende gennemsnit for hver team
        df_packing_time['rolling_packing'] = df_packing_time.groupby('team_name')['packing_diff'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        
        fig1 = go.Figure()

        for team in df_packing_time['team_name'].unique():
            team_data = df_packing_time[df_packing_time['team_name'] == team]
            line_size = 5 if team == 'Horsens' else 1  # Større linje for Horsens
            fig1.add_trace(go.Scatter(
                x=team_data['date'],
                y=team_data['rolling_packing'],
                mode='lines',
                name=team,
                line=dict(width=line_size)
            ))

        fig1.update_layout(
            title='3-Game Rolling Average of packing difference',
            xaxis_title='Date',
            yaxis_title='3-Game Rolling Average of packing difference',
            template='plotly_white'
        )
        st.plotly_chart(fig1)
        
        df_packing_period = df_packing[df_packing['label'].isin(match_choice)]
        df_packing_period = df_packing_period[['label', 'team_name', 'bypassed_opponents', 'bypassed_defenders']]
        df_packing_period = df_packing_period.groupby(['label', 'team_name'])[['bypassed_opponents','bypassed_defenders']].sum().reset_index()
        df_packing_period = df_packing_period.sort_values(by='bypassed_opponents', ascending=False)
        df_packing_period['packing_match'] = df_packing_period.groupby('label')['bypassed_opponents'].transform('sum')
        df_packing_period['packing_diff'] = df_packing_period['bypassed_opponents'] - df_packing_period['packing_match'] + df_packing_period['bypassed_opponents']
        df_packing_period = df_packing_period[df_packing_period['team_name'] == 'Horsens']
        df_packing_period = df_packing_period[['label','bypassed_opponents', 'packing_diff']]
        
        st.header('Chosen matches')
        st.dataframe(df_packing_period, hide_index=True)
        
        df_packing_pass_received_player = df_packing[df_packing['label'].isin(match_choice)]
        df_packing_pass_received_player = df_packing_pass_received_player[df_packing_pass_received_player['team_name'] == 'Horsens']
        df_packing_pass_received_player = df_packing_pass_received_player[['pass_receiver', 'bypassed_opponents']]
        df_packing_pass_received_player = df_packing_pass_received_player.groupby(['pass_receiver'])['bypassed_opponents'].sum().reset_index()
        df_packing_pass_received_player = df_packing_pass_received_player.sort_values(by='bypassed_opponents', ascending=False)
        df_packing_pass_received_player.rename(columns={'pass_receiver': 'playerName', 'bypassed_opponents': 'bypassed_opponents_received'}, inplace=True)
        
        df_packing_period_player = df_packing[df_packing['label'].isin(match_choice)]
        df_packing_period_player = df_packing_period_player[df_packing_period_player['team_name'] == 'Horsens']
        df_packing_period_player = df_packing_period_player[['playerName', 'bypassed_opponents', 'bypassed_defenders']]
        df_packing_period_player = df_packing_period_player.groupby(['playerName'])[['bypassed_opponents','bypassed_defenders']].sum().reset_index()
        df_packing_period_player = df_packing_period_player.sort_values(by='bypassed_opponents', ascending=False)
        df_packing_period_player = df_packing_period_player.merge(df_packing_pass_received_player, on='playerName', how='left')
        df_packing_period_player.rename(columns={'bypassed_opponents': 'packing', 'bypassed_defenders': 'packing_defenders', 'bypassed_opponents_received': 'packing_received'}, inplace=True)
        
        st.dataframe(df_packing_period_player, hide_index=True)
        
        df_packing_first_third = df_packing[df_packing['label'].isin(match_choice)]
        df_packing_first_third = df_packing_first_third[df_packing_first_third['x'] <= 33.3]
        df_packing_first_third = df_packing_first_third[df_packing_first_third['team_name'] == 'Horsens']   
        df_packing_first_third = df_packing_first_third[df_packing_first_third['bypassed_opponents'] > 0]
        df_packing_first_third = df_packing_first_third[['closest_opponent_distance']]
        fig_histogram = px.histogram(df_packing_first_third, x='closest_opponent_distance', nbins=30, title='Histogram of Closest Opponent Distance')
        st.plotly_chart(fig_histogram)

    def chance_creation():
        df_matchstats = load_match_stats(columns=['contestantId','date', 'label', 'touchesInOppBox'])
        df_matchstats['date'] = pd.to_datetime(df_matchstats['date'])
        df_xA = load_xA()
        df_crosses = load_crosses()
        xA_map = df_xA[['contestantId', 'team_name']].drop_duplicates()
        df_matchstats = df_matchstats.merge(xA_map, on='contestantId')
        
        df_possession = load_possession_data()
        df_possession = df_possession[~(df_possession[['6.0','107.0']] == True).any(axis=1)]
        df_possession = df_possession[df_possession['label'].isin(match_choice)]
        df_possession = df_possession[df_possession['team_name'] == 'Horsens']

        df_possession = df_possession.sort_values(by='eventId').reset_index(drop=True)        
        df_possession['pass_receiver'] = None
        for i in range(len(df_possession) - 1):
            current_event = df_possession.loc[i]
            if current_event['typeId'] == 1 and current_event['outcome'] == 1:
                next_event_id = current_event['eventId'] + 1
                next_event = df_possession[(df_possession['eventId'] == next_event_id) & (df_possession['team_name'] == current_event['team_name'])]

                if not next_event.empty:
                    pass_receiver = next_event.iloc[0]['playerName']
                    df_possession.at[i, 'pass_receiver'] = pass_receiver

        df_passes = df_possession[df_possession['team_name'] == 'Horsens']
        df_passes = df_passes[df_passes['label'].isin(match_choice)]
        

        df_forward_passes = df_passes[df_passes['typeId'] == 1]
        df_passes = df_passes[(df_passes['typeId'] == 1) & (df_passes['outcome'] == 1)]
        assistzone_pass_ends = df_passes[
            (df_passes['140.0'].astype(float) >= 83) &
            (df_passes['141.0'].astype(float) >= 21.1) & 
            (df_passes['141.0'].astype(float) <= 36.8)|
            (df_passes['140.0'].astype(float) >= 83) &
            (df_passes['141.0'].astype(float) >= 63.2) &
            (df_passes['141.0'].astype(float) <= 78.9)
        ]

        team_counts = assistzone_pass_ends.groupby(['team_name','label']).size().reset_index(name='count')
        team_counts.columns = ['team_name', 'label', 'count']
        team_counts = team_counts.sort_values(by=['count'], ascending=False)

        # Tæl forekomster af hver playerName
        player_counts = assistzone_pass_ends['playerName'].value_counts().reset_index(name='Passed')
        player_counts.columns = ['playerName', 'Passed']
        pass_receiver_counts = assistzone_pass_ends['pass_receiver'].value_counts().reset_index(name='Received')
        pass_receiver_counts.columns = ['pass_receiver', 'Received']
        pass_receiver_counts.rename(columns={'pass_receiver': 'playerName'}, inplace=True)
        player_counts = player_counts.merge(pass_receiver_counts, on='playerName', how='outer')
        player_counts.fillna(0, inplace=True)
        player_counts['Total'] = player_counts['Passed'] + player_counts['Received']
        player_counts = player_counts.sort_values(by=['Total'], ascending=False)
        
        st.header('Passes into halfspace in the box')
        st.dataframe(player_counts,hide_index=True)
        st.dataframe(team_counts,hide_index=True)

        
        option = st.selectbox(
            'Select the position to display',
            ('Start', 'End')
        )

        # Initialize the pitch
        pitch = Pitch(pitch_type='opta',line_zorder=2, pitch_color='grass', line_color='white')
        fig, ax = pitch.draw()

        # Extract coordinates based on user selection
        if option == 'Start':
            x_coords = assistzone_pass_ends['x']
            y_coords = assistzone_pass_ends['y']
        else:
            x_coords = assistzone_pass_ends['140.0']
            y_coords = assistzone_pass_ends['141.0']

        # Plot the heatmap
        fig.set_facecolor('#22312b')
        bin_statistic = pitch.bin_statistic(x_coords, y_coords, statistic='count', bins=(50, 50)) # Adjust bins as needed
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')

        pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='black')

        # Display the plot in Streamlit
        st.pyplot(fig)

        st.header('Touches in zone 14')
        df_zone14 = df_possession[(df_possession['x'].astype(float) >= 66) & ((df_possession['y'].astype(float) >= 21.1) & (df_possession['y'].astype(float) <= 78.9))]
        
        df_zone14_team = df_zone14.groupby(['team_name', 'label']).size().reset_index(name='Touches')
        df_zone14_team = df_zone14_team.sort_values(by=['Touches'], ascending=False)
        df_zone14_player = df_zone14.groupby(['playerName']).size().reset_index(name='Touches')
        df_zone14_player = df_zone14_player.sort_values(by=['Touches'], ascending=False)
        st.dataframe(df_zone14_team,hide_index=True)
        st.dataframe(df_zone14_player, hide_index=True)
        
        st.header('Touches in box')
        st.write('Whole season')
        touches_in_box_player = df_matchstats[df_matchstats['team_name'] == 'Horsens']
        touches_in_box_player = touches_in_box_player[touches_in_box_player['label'].isin(match_choice)]
        touches_in_box_player = touches_in_box_player.groupby(['player_matchName'])['touchesInOppBox'].sum().reset_index()
        touches_in_box_player = touches_in_box_player.sort_values(by=['touchesInOppBox'], ascending=False)
        touches_in_box_team = df_matchstats.groupby(['team_name','date', 'label'])['touchesInOppBox'].sum().reset_index()
        touches_in_box_team['tib_match'] = touches_in_box_team.groupby('label')['touchesInOppBox'].transform('sum')
        touches_in_box_team['touches_in_box_diff'] = touches_in_box_team['touchesInOppBox'] - touches_in_box_team['tib_match'] + touches_in_box_team['touchesInOppBox']
        touches_in_box_team = touches_in_box_team.sort_values(by=['date'], ascending=True)
        touches_in_box_team['rolling_touches_in_box'] = touches_in_box_team.groupby('team_name')['touches_in_box_diff'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        touches_in_box_team_period = touches_in_box_team[touches_in_box_team['label'].isin(match_choice)]
        touches_in_box_team_period = touches_in_box_team_period[touches_in_box_team_period['team_name'] == 'Horsens']
        touches_in_box_team_period = touches_in_box_team_period[['team_name','label', 'touches_in_box_diff']]
        touches_in_box_team_period = touches_in_box_team_period.sort_values(by=['touches_in_box_diff'], ascending=False)
        fig1 = go.Figure()

        for team in touches_in_box_team['team_name'].unique():
            team_data = touches_in_box_team[touches_in_box_team['team_name'] == team]
            line_size = 5 if team == 'Horsens' else 1  # Larger line for Horsens
            fig1.add_trace(go.Scatter(
                x=team_data['date'],
                y=team_data['rolling_touches_in_box'],
                mode='lines',
                name=team,
                line=dict(width=line_size)
            ))

        fig1.update_layout(
            title='3-Game Rolling Average of touches in box difference',
            xaxis_title='Date',
            yaxis_title='3-Game Rolling Average of touches in box difference',
            template='plotly_white'
        )

        st.plotly_chart(fig1)
        st.write('Chosen matches')
        st.dataframe(touches_in_box_team_period, hide_index=True)
        st.dataframe(touches_in_box_player, hide_index=True)      

    def crosses():
        df_crosses = load_crosses()
        df_crosses = df_crosses[df_crosses['label'].isin(match_choice)]
        df_crosses['qualifier'] = df_crosses['qualifier'].apply(ast.literal_eval)

        # List of qualifierIds to filter out
        filter_qualifier_ids = [5, 6, 24, 25, 26, 107]

        # Function to check if any dictionary in the list has a qualifierId in the filter list
        def filter_qualifiers(qualifier_list):
            return not any(d['qualifierId'] in filter_qualifier_ids for d in qualifier_list)

        # Filter the DataFrame
        df_crosses = df_crosses[df_crosses['qualifier'].apply(filter_qualifiers)]
        def early_crosses(df_crosses):
            st.header('Early crosses')
            df_early_crosses = df_crosses[(df_crosses['x'].astype(float) <= 88.5) &(df_crosses['x'].astype(float) >= 70.0) & ((df_crosses['y'].astype(float) >= 78.9) | (df_crosses['y'].astype(float) <= 21.1))]
            
            pitch = Pitch(pitch_type='opta', half=True,pitch_color='grass')  # Create a half-pitch plot
            fig, ax = pitch.draw(figsize=(10, 8))

            # Plot arrows from x,y to pass_end_x,pass_end_y
            for _, row in df_early_crosses.iterrows():
                pitch.arrows(row['x'], row['y'], row['pass_end_x'], row['pass_end_y'],
                            color='blue', ax=ax, width=2, headwidth=5, headlength=5, headaxislength=4.5)

            # Add labels for players (optional)
            for _, row in df_early_crosses.iterrows():
                ax.text(row['x'], row['y'], row['playerName'], fontsize=12, color='black')

            # Display the plot in Streamlit
            st.pyplot(fig)

            def parse_players(players):
                try:
                    return ast.literal_eval(players) if isinstance(players, str) else players
                except (ValueError, SyntaxError):
                    return []  # Return an empty list if parsing fails

            def count_teammates_near_goal(teammates, distance_threshold=20):
                count = 0
                player_names_near_goal = []
                
                for teammate in teammates:
                    distance_to_opponents_goal = teammate.get('distance_to_opponents_goal', None)
                    if distance_to_opponents_goal is not None:
                        if distance_to_opponents_goal <= distance_threshold:
                            count += 1
                            player_names_near_goal.append(teammate['name'])
                
                return count, player_names_near_goal

            # Initialize the new column with default values (e.g., 0)
            df_early_crosses['#players in box'] = 0
            df_early_crosses['players in box'] = ''

            # Loop through the DataFrame
            for idx, row in df_early_crosses.iterrows():
                player_name = row['playerName']
                start_homePlayers = parse_players(row['start_homePlayers'])
                start_awayPlayers = parse_players(row['start_awayPlayers'])
                end_homePlayers = parse_players(row['end_homePlayers'])
                end_awayPlayers = parse_players(row['end_awayPlayers'])
                
                # Ensure teammates is always a list
                teammates = []

                # Determine if the player is in homePlayers or awayPlayers
                if isinstance(end_homePlayers, list) and player_name in [player['name'] for player in end_homePlayers]:
                    teammates = end_homePlayers
                elif isinstance(start_homePlayers, list) and player_name in [player['name'] for player in start_homePlayers]:
                    teammates = start_homePlayers
                elif isinstance(end_awayPlayers, list) and player_name in [player['name'] for player in end_awayPlayers]:
                    teammates = end_awayPlayers
                elif isinstance(start_awayPlayers, list) and player_name in [player['name'] for player in start_awayPlayers]:
                    teammates = start_awayPlayers

                if isinstance(teammates, list):
                    # Count teammates near opponents' goal and get their names
                    num_teammates_near_goal, player_names_near_goal = count_teammates_near_goal(teammates)
                    df_early_crosses.at[idx, '#players in box'] = num_teammates_near_goal
                    df_early_crosses.at[idx, 'players in box'] = ', '.join(player_names_near_goal)

            fig_histogram = px.histogram(df_early_crosses, x='#players in box', nbins=30, title='#Players in box')
            st.plotly_chart(fig_histogram)        
            def count_players_in_box(player_lists):
                # Flatten the list of strings into a single list of player names
                all_players = [player.strip() for sublist in player_lists for player in sublist.split(",")]
                
                # Count the occurrences of each player's name
                player_counts = Counter(all_players)
                
                return player_counts

            player_lists = df_early_crosses['players in box'].tolist()
            player_counts = count_players_in_box(player_lists)
            st.write("Player Counts in the Box:")
            df_player_counts = pd.DataFrame(player_counts.items(), columns=['Player', 'Times in Box'])
            df_player_counts = df_player_counts.sort_values(by=['Times in Box'], ascending=False)
            df_player_counts = df_player_counts[df_player_counts['Player'] != '']
            st.dataframe(df_player_counts, hide_index=True)
        early_crosses(df_crosses)
        
        def late_crosses(df_crosses):
            st.header('Late crosses')
            df_early_crosses = df_crosses[(df_crosses['x'].astype(float) > 88.5) & ((df_crosses['y'].astype(float) >= 78.9) | (df_crosses['y'].astype(float) <= 21.1))]
            
            pitch = Pitch(pitch_type='opta', half=True,pitch_color='grass')  # Create a half-pitch plot
            fig, ax = pitch.draw(figsize=(10, 8))

            # Plot arrows from x,y to pass_end_x,pass_end_y
            for _, row in df_early_crosses.iterrows():
                pitch.arrows(row['x'], row['y'], row['pass_end_x'], row['pass_end_y'],
                            color='blue', ax=ax, width=2, headwidth=5, headlength=5, headaxislength=4.5)

            # Add labels for players (optional)
            for _, row in df_early_crosses.iterrows():
                ax.text(row['x'], row['y'], row['playerName'], fontsize=12, color='black')


            # Display the plot in Streamlit
            st.pyplot(fig)

            def parse_players(players):
                try:
                    return ast.literal_eval(players) if isinstance(players, str) else players
                except (ValueError, SyntaxError):
                    return []  # Return an empty list if parsing fails

            def count_teammates_near_goal(teammates, distance_threshold=20):
                count = 0
                player_names_near_goal = []
                
                for teammate in teammates:
                    distance_to_opponents_goal = teammate.get('distance_to_opponents_goal', None)
                    if distance_to_opponents_goal is not None:
                        if distance_to_opponents_goal <= distance_threshold:
                            count += 1
                            player_names_near_goal.append(teammate['name'])
                
                return count, player_names_near_goal

            # Initialize the new column with default values (e.g., 0)
            df_early_crosses['#players in box'] = 0
            df_early_crosses['players in box'] = ''

            # Loop through the DataFrame
            for idx, row in df_early_crosses.iterrows():
                player_name = row['playerName']
                start_homePlayers = parse_players(row['start_homePlayers'])
                start_awayPlayers = parse_players(row['start_awayPlayers'])
                end_homePlayers = parse_players(row['end_homePlayers'])
                end_awayPlayers = parse_players(row['end_awayPlayers'])
                
                # Ensure teammates is always a list
                teammates = []

                # Determine if the player is in homePlayers or awayPlayers
                if isinstance(end_homePlayers, list) and player_name in [player['name'] for player in end_homePlayers]:
                    teammates = end_homePlayers
                elif isinstance(start_homePlayers, list) and player_name in [player['name'] for player in start_homePlayers]:
                    teammates = start_homePlayers
                elif isinstance(end_awayPlayers, list) and player_name in [player['name'] for player in end_awayPlayers]:
                    teammates = end_awayPlayers
                elif isinstance(start_awayPlayers, list) and player_name in [player['name'] for player in start_awayPlayers]:
                    teammates = start_awayPlayers

                if isinstance(teammates, list):
                    # Count teammates near opponents' goal and get their names
                    num_teammates_near_goal, player_names_near_goal = count_teammates_near_goal(teammates)
                    df_early_crosses.at[idx, '#players in box'] = num_teammates_near_goal
                    df_early_crosses.at[idx, 'players in box'] = ', '.join(player_names_near_goal)

            fig_histogram = px.histogram(df_early_crosses, x='#players in box', nbins=30, title='#Players in box')
            st.plotly_chart(fig_histogram)        
            def count_players_in_box(player_lists):
                # Flatten the list of strings into a single list of player names
                all_players = [player.strip() for sublist in player_lists for player in sublist.split(",")]
                
                # Count the occurrences of each player's name
                player_counts = Counter(all_players)
                
                return player_counts

            player_lists = df_early_crosses['players in box'].tolist()
            player_counts = count_players_in_box(player_lists)
            st.write("Player Counts in the Box:")
            df_player_counts = pd.DataFrame(player_counts.items(), columns=['Player', 'Times in Box'])
            df_player_counts = df_player_counts.sort_values(by=['Times in Box'], ascending=False)
            df_player_counts = df_player_counts[df_player_counts['Player'] != '']
            st.dataframe(df_player_counts, hide_index=True)
        late_crosses(df_crosses)
        
        def cutback(df_crosses):
            st.header('Cutbacks')
            df_early_crosses = df_crosses[
                (df_crosses['x'].astype(float) > 83.0) & 
                (
                    ((df_crosses['y'].astype(float) <= 78.9) & (df_crosses['y'].astype(float) >= 63.2)) | 
                    ((df_crosses['y'].astype(float) >= 21.1) & (df_crosses['y'].astype(float) <= 36.8))
                )
            ]
            
            pitch = Pitch(pitch_type='opta', half=True,pitch_color='grass')  # Create a half-pitch plot
            fig, ax = pitch.draw(figsize=(10, 8))

            # Plot arrows from x,y to pass_end_x,pass_end_y
            for _, row in df_early_crosses.iterrows():
                pitch.arrows(row['x'], row['y'], row['pass_end_x'], row['pass_end_y'],
                            color='blue', ax=ax, width=2, headwidth=5, headlength=5, headaxislength=4.5)

            # Add labels for players (optional)
            for _, row in df_early_crosses.iterrows():
                ax.text(row['x'], row['y'], row['playerName'], fontsize=12, color='black')

            plt.title('Cutbacks', fontsize=20)

            # Display the plot in Streamlit
            st.pyplot(fig)

            def parse_players(players):
                try:
                    return ast.literal_eval(players) if isinstance(players, str) else players
                except (ValueError, SyntaxError):
                    return []  # Return an empty list if parsing fails

            def count_teammates_near_goal(teammates, distance_threshold=20):
                count = 0
                player_names_near_goal = []
                
                for teammate in teammates:
                    distance_to_opponents_goal = teammate.get('distance_to_opponents_goal', None)
                    if distance_to_opponents_goal is not None:
                        if distance_to_opponents_goal <= distance_threshold:
                            count += 1
                            player_names_near_goal.append(teammate['name'])
                
                return count, player_names_near_goal

            # Initialize the new column with default values (e.g., 0)
            df_early_crosses['#players in box'] = 0
            df_early_crosses['players in box'] = ''

            # Loop through the DataFrame
            for idx, row in df_early_crosses.iterrows():
                player_name = row['playerName']
                start_homePlayers = parse_players(row['start_homePlayers'])
                start_awayPlayers = parse_players(row['start_awayPlayers'])
                end_homePlayers = parse_players(row['end_homePlayers'])
                end_awayPlayers = parse_players(row['end_awayPlayers'])
                
                # Ensure teammates is always a list
                teammates = []

                # Determine if the player is in homePlayers or awayPlayers
                if isinstance(end_homePlayers, list) and player_name in [player['name'] for player in end_homePlayers]:
                    teammates = end_homePlayers
                elif isinstance(start_homePlayers, list) and player_name in [player['name'] for player in start_homePlayers]:
                    teammates = start_homePlayers
                elif isinstance(end_awayPlayers, list) and player_name in [player['name'] for player in end_awayPlayers]:
                    teammates = end_awayPlayers
                elif isinstance(start_awayPlayers, list) and player_name in [player['name'] for player in start_awayPlayers]:
                    teammates = start_awayPlayers

                if isinstance(teammates, list):
                    # Count teammates near opponents' goal and get their names
                    num_teammates_near_goal, player_names_near_goal = count_teammates_near_goal(teammates)
                    df_early_crosses.at[idx, '#players in box'] = num_teammates_near_goal
                    df_early_crosses.at[idx, 'players in box'] = ', '.join(player_names_near_goal)

            fig_histogram = px.histogram(df_early_crosses, x='#players in box', nbins=30, title='#Players in box')
            st.plotly_chart(fig_histogram)        
            def count_players_in_box(player_lists):
                # Flatten the list of strings into a single list of player names
                all_players = [player.strip() for sublist in player_lists for player in sublist.split(",")]
                
                # Count the occurrences of each player's name
                player_counts = Counter(all_players)
                
                return player_counts

            player_lists = df_early_crosses['players in box'].tolist()
            player_counts = count_players_in_box(player_lists)
            st.write("Player Counts in the Box:")
            df_player_counts = pd.DataFrame(player_counts.items(), columns=['Player', 'Times in Box'])
            df_player_counts = df_player_counts.sort_values(by=['Times in Box'], ascending=False)
            df_player_counts = df_player_counts[df_player_counts['Player'] != '']
            st.dataframe(df_player_counts, hide_index=True)
        cutback(df_crosses)

    def pressing():
        df_possession_data = load_possession_data()
        def calculate_ppda(df_possession_data):
            df_ppda = df_possession_data[df_possession_data['typeId'].isin([1, 4, 7,8, 45])]
            df_ppdabeyond40 = df_ppda[df_ppda['x'].astype(float) > 40]
            df_ppdabeyond40_passes = df_ppdabeyond40[df_ppdabeyond40['typeId'] == 1]
            df_ppdabeyond40_passestotal = df_ppdabeyond40_passes.groupby(['label','date'])['eventId'].count().reset_index()
            df_ppdabeyond40_passestotal = df_ppdabeyond40_passestotal.rename(columns={'eventId': 'passes in game'})
            df_ppdabeyond40_passesteams = df_ppdabeyond40_passes.groupby(['label','team_name','date'])['eventId'].count().reset_index()
            df_ppdabeyond40_passesteams = df_ppdabeyond40_passesteams.rename(columns={'eventId': 'passes'})

            df_ppdabeyond40_defactions = df_ppdabeyond40[df_ppdabeyond40['typeId'].isin([4, 7, 8, 45])]
            df_ppdabeyond40_defactionstotal = df_ppdabeyond40_defactions.groupby(['label','date'])['eventId'].count().reset_index()
            df_ppdabeyond40_defactionstotal = df_ppdabeyond40_defactionstotal.rename(columns={'eventId': 'defensive actions in game'})
            df_ppdabeyond40_defactionsteams = df_ppdabeyond40_defactions.groupby(['label', 'team_name','date'])['eventId'].count().reset_index()
            df_ppdabeyond40_defactionsteams = df_ppdabeyond40_defactionsteams.rename(columns={'eventId': 'defensive actions'})
            df_ppdabeyond40total = df_ppdabeyond40_defactionstotal.merge(df_ppdabeyond40_passestotal)
            df_ppdabeyond40 = df_ppdabeyond40_defactionsteams.merge(df_ppdabeyond40total)
            df_ppdabeyond40 = df_ppdabeyond40.merge(df_ppdabeyond40_passesteams)
            df_ppdabeyond40['opponents passes'] = df_ppdabeyond40['passes in game'] - df_ppdabeyond40['passes']
            df_ppdabeyond40['PPDA'] = df_ppdabeyond40['opponents passes'] / df_ppdabeyond40['defensive actions']
            df_ppda = df_ppdabeyond40[['label', 'team_name','date', 'PPDA']]
            df_ppda = df_ppda.sort_values(by=['date'], ascending=True)
            return df_ppda
        
        df_ppda = calculate_ppda(df_possession_data)
        df_ppda = df_ppda[df_ppda['team_name'] == 'Horsens']
        df_ppda_season_average = df_ppda.groupby(['team_name'])['PPDA'].mean().reset_index()
        average_ppda = df_ppda_season_average['PPDA'][0]
        df_ppda_sorted = df_ppda.sort_values(by=['date', 'label'], ascending=[True, True])
        df_counterpressing = counterpressing()
        df_counterpressing = df_counterpressing.sort_values(by=['date'], ascending=True)

        def add_avg_line(fig, avg):
            fig.add_shape(
                type="line",
                x0=-0.5, x1=len(fig.data[0].x)-0.5,
                y0=avg, y1=avg,
                line=dict(color="Red", dash="dash")
            )
            fig.add_annotation(
                x=len(fig.data[0].x)-0.5,
                y=avg,
                text=f"Avg PPDA: {avg:.2f}",
                showarrow=False,
                yshift=10
        )
            
        st.header('Whole season')
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df_counterpressing['label'],
            y=df_counterpressing['counterpressing_5s'],
            name='Counterpressing 5s'
        ))

        fig.add_trace(go.Bar(
            x=df_counterpressing['label'],
            y=df_counterpressing['counterpressing_10s'],
            name='Counterpressing 10s',
            base=df_counterpressing['counterpressing_5s']
        ))

        fig.update_layout(
            barmode='stack',
            title='Counterpressing Events',
            legend_title='Event Type'
        )

        st.plotly_chart(fig)

        
        fig_whole_season = px.bar(df_ppda_sorted, x='label', y='PPDA', title='PPDA for Horsens - Whole Season')
        add_avg_line(fig_whole_season, average_ppda)
        st.plotly_chart(fig_whole_season)

        st.header('Chosen matches')
        df_counterpressing = df_counterpressing[df_counterpressing['label'].isin(match_choice)]
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df_counterpressing['label'],
            y=df_counterpressing['counterpressing_5s'],
            name='Counterpressing 5s'
        ))

        fig.add_trace(go.Bar(
            x=df_counterpressing['label'],
            y=df_counterpressing['counterpressing_10s'],
            name='Counterpressing 10s',
            base=df_counterpressing['counterpressing_5s']
        ))

        fig.update_layout(
            barmode='stack',
            title='Counterpressing Events',
            xaxis_title='Match',
            yaxis_title='Number of Events',
            legend_title='Event Type'
        )

        st.plotly_chart(fig)
        df_ppda_chosen_period = df_ppda_sorted[df_ppda_sorted['label'].isin(match_choice)]
        fig_chosen_matches = px.bar(df_ppda_chosen_period, x='label', y='PPDA', title='PPDA for Horsens - Chosen Matches')
        add_avg_line(fig_chosen_matches, average_ppda)
        st.plotly_chart(fig_chosen_matches)
        
    Data_types = {
        'xG': xg,
        'Passing':passes,
        'Packing': packing,
        'Chance Creation': chance_creation,
        'Pressing': pressing,
        'Crosses': crosses
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

def League_stats():
    
    
    balanced_central_defender_df = position_dataframes['Central defender']
    fullbacks_df = position_dataframes['Fullbacks']
    number6_df = position_dataframes['Number 6']
    number8_df = position_dataframes['Number 8']
    number10_df = position_dataframes['Number 10']
    winger_df = position_dataframes['Winger']
    classic_striker_df = position_dataframes['Classic striker']

    matchstats_df = load_match_stats(columns = any)
    matchstats_df = matchstats_df.rename(columns={'player_matchName': 'playerName'})
    matchstats_df = matchstats_df.groupby(['contestantId','label', 'date']).sum().reset_index()
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
        value=(default_start_date, default_end_date)
    )

    selected_start_date = pd.to_datetime(selected_start_date, format=date_format)
    selected_end_date = pd.to_datetime(selected_end_date, format=date_format)
    filtered_data = matchstats_df[
        (matchstats_df['date'] >= selected_start_date) & (matchstats_df['date'] <= selected_end_date)
    ]    
    
    xg_df = load_all_xg()
    xg_df_openplay = xg_df[xg_df['321'] > 0]

    xg_df_openplay = xg_df_openplay.groupby(['contestantId', 'team_name', 'date'])['321'].sum().reset_index()
    xg_df_openplay = xg_df_openplay.rename(columns={'321': 'open play xG'})
    xg_df_openplay['date'] = pd.to_datetime(xg_df_openplay['date'])
    
    df_spacecontrol = load_spacecontrol_data()
    df_spacecontrol = df_spacecontrol[df_spacecontrol['Type'] == 'Player']
    df_spacecontrol = df_spacecontrol[['Team','date','TotalControlArea','CenterControlArea','PenaltyAreaControl','label']]
    df_spacecontrol[['TotalControlArea', 'CenterControlArea', 'PenaltyAreaControl']] = df_spacecontrol[['TotalControlArea', 'CenterControlArea', 'PenaltyAreaControl']].astype(float).round(2)
    
    df_spacecontrol = df_spacecontrol.groupby(['Team','label','date']).sum().reset_index()
    df_spacecontrol['date'] = pd.to_datetime(df_spacecontrol['date'])
    df_spacecontrol['TotalControlArea_match'] = df_spacecontrol.groupby('label')['TotalControlArea'].transform('sum')
    df_spacecontrol['CenterControlArea_match'] = df_spacecontrol.groupby('label')['CenterControlArea'].transform('sum')
    df_spacecontrol['PenaltyAreaControl_match'] = df_spacecontrol.groupby('label')['PenaltyAreaControl'].transform('sum')

    df_spacecontrol['Total Control Area %'] = df_spacecontrol['TotalControlArea'] / df_spacecontrol['TotalControlArea_match'] * 100
    df_spacecontrol['Center Control Area %'] = df_spacecontrol['CenterControlArea'] / df_spacecontrol['CenterControlArea_match'] * 100
    df_spacecontrol['Penalty Area Control %'] = df_spacecontrol['PenaltyAreaControl'] / df_spacecontrol['PenaltyAreaControl_match'] * 100


    df_spacecontrol = df_spacecontrol[['Team','date', 'Total Control Area %', 'Center Control Area %', 'Penalty Area Control %']]
    df_spacecontrol = df_spacecontrol.rename(columns={'Team': 'team_name'})
    df_spacecontrol['Total Control Area %'] = df_spacecontrol['Total Control Area %'].round(2)
    df_spacecontrol['Center Control Area %'] = df_spacecontrol['Center Control Area %'].round(2)
    df_spacecontrol['Penalty Area Control %'] = df_spacecontrol['Penalty Area Control %'].round(2)
    df_ppda = load_ppda()
    df_ppda = df_ppda.groupby(['team_name','date']).sum().reset_index()
    df_ppda['date'] = pd.to_datetime(df_ppda['date'])
    df_ppda['PPDA'] = df_ppda['PPDA'].astype(float).round(2)
    df_ppda = df_ppda[['team_name','date', 'PPDA']]
    matchstats_df = xg_df_openplay.merge(filtered_data)
    matchstats_df = df_ppda.merge(matchstats_df)

    matchstats_df = matchstats_df.merge(df_spacecontrol,how='left')


    matchstats_df = matchstats_df.drop(columns='date')
    # Perform aggregation
    matchstats_df = matchstats_df.groupby(['contestantId', 'team_name']).agg({
        'label': 'sum',  # Example of a column to sum
        'penAreaEntries': 'sum',  # Example of another column to sum
        'open play xG': 'sum',  # Example of a column to average
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
        'Total Control Area %': 'mean',
        'Center Control Area %': 'mean',
        'Penalty Area Control %': 'mean',
        'PPDA': 'mean',
        }).reset_index()
    
    matchstats_df = matchstats_df.rename(columns={'label': 'matches'})
    matchstats_df['PenAreaEntries per match'] = matchstats_df['penAreaEntries'] / matchstats_df['matches']
    matchstats_df['Open play xG per match'] = matchstats_df['open play xG'] / matchstats_df['matches']
    matchstats_df['Duels per match'] = (matchstats_df['duelLost'] + matchstats_df['duelWon']) /matchstats_df['matches']
    matchstats_df['Duels won %'] = matchstats_df['duelWon'] / (matchstats_df['duelWon'] + matchstats_df['duelLost'])	
    matchstats_df['Passes per game'] = matchstats_df['openPlayPass'] / matchstats_df['matches']
    matchstats_df['Pass accuracy %'] = matchstats_df['successfulOpenPlayPass'] / matchstats_df['openPlayPass']
    matchstats_df['Back zone pass accuracy %'] = matchstats_df['accurateBackZonePass'] / matchstats_df['totalBackZonePass']
    matchstats_df['Forward zone pass accuracy %'] = matchstats_df['accurateFwdZonePass'] / matchstats_df['totalFwdZonePass']
    matchstats_df['possWonDef3rd %'] = matchstats_df['possWonDef3rd'] / (matchstats_df['possWonDef3rd'] + matchstats_df['possWonMid3rd'] + matchstats_df['possWonAtt3rd'])    
    matchstats_df['possWonMid3rd %'] = matchstats_df['possWonMid3rd'] / (matchstats_df['possWonDef3rd'] + matchstats_df['possWonMid3rd'] + matchstats_df['possWonAtt3rd'])    
    matchstats_df['possWonAtt3rd %'] = matchstats_df['possWonAtt3rd'] / (matchstats_df['possWonDef3rd'] + matchstats_df['possWonMid3rd'] + matchstats_df['possWonAtt3rd'])    
    matchstats_df['Forward pass share %'] = matchstats_df['fwdPass'] / matchstats_df['openPlayPass']
    matchstats_df['Final third entries per match'] = matchstats_df['finalThirdEntries'] / matchstats_df['matches']
    matchstats_df['Final third pass accuracy %'] = matchstats_df['successfulFinalThirdPasses'] / matchstats_df['totalFinalThirdPasses']
    matchstats_df['Open play shot assists share'] = matchstats_df['attAssistOpenplay'] / matchstats_df['totalAttAssist']
    matchstats_df['Long pass share %'] = matchstats_df['totalLongBalls'] / matchstats_df['openPlayPass']
    matchstats_df['Crosses'] = matchstats_df['totalCrossNocorner']
    matchstats_df['Cross accuracy %'] = matchstats_df['accurateCrossNocorner'] / matchstats_df['totalCrossNocorner']
    matchstats_df['PPDA per match'] = matchstats_df['PPDA']
    matchstats_df['Total Space control %'] = matchstats_df['Total Control Area %']
    matchstats_df['Center Space control %'] = matchstats_df['Center Control Area %']
    matchstats_df['Penalty Area Space control %'] = matchstats_df['Penalty Area Control %']
    matchstats_df = matchstats_df[['team_name','matches','PenAreaEntries per match','Open play xG per match','Duels per match','Duels won %','Passes per game','Pass accuracy %','Back zone pass accuracy %','Forward zone pass accuracy %','possWonDef3rd %','possWonMid3rd %','possWonAtt3rd %','Forward pass share %','Final third entries per match','Final third pass accuracy %','Open play shot assists share','PPDA per match','Total Space control %','Center Space control %','Penalty Area Space control %','Long pass share %','Crosses','Cross accuracy %']]

    cols_to_rank = matchstats_df.drop(columns=['team_name']).columns
    ranked_df = matchstats_df.copy()  # Create a copy of the original DataFrame
    for col in cols_to_rank:
        # Special case for 'PPDA per match' to rank in descending order
        if col == 'PPDA per match':
            ranked_df[col + '_rank'] = matchstats_df[col].rank(axis=0, ascending=True)
        else:
            ranked_df[col + '_rank'] = matchstats_df[col].rank(axis=0, ascending=False)
    matchstats_df = ranked_df.merge(matchstats_df)
    matchstats_df = matchstats_df.set_index('team_name')
    matchstats_df = matchstats_df.drop(columns=['matches_rank'])
    st.dataframe(matchstats_df)
    matchstats_df = matchstats_df.reset_index()

    # Select a team
    sorted_teams = matchstats_df['team_name'].sort_values()

    # Select a team
    selected_team = st.selectbox('Choose team', sorted_teams)
    team_df = matchstats_df.loc[matchstats_df['team_name'] == selected_team]

    # Target ranks
    target_ranks = [1, 2, 3, 4, 9, 10, 11, 12]

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

    # Find similar teams
    selected_columns = [
        'Duels per match_rank', 
        'Duels won %_rank', 'Passes per game_rank', 'Pass accuracy %_rank', 'Back zone pass accuracy %_rank',
        'Forward zone pass accuracy %_rank', 'possWonDef3rd %_rank', 'possWonMid3rd %_rank', 
        'possWonAtt3rd %_rank', 'Forward pass share %_rank', 
        'Final third pass accuracy %_rank', 'Open play shot assists share_rank', 'PPDA per match_rank', 
        'Long pass share %_rank', 'Crosses_rank', 'Cross accuracy %_rank'
    ]

    rank_columns = [col for col in matchstats_df.columns if col.endswith('_rank')]

    matchstats_df['similarity_score'] = matchstats_df.apply(
        lambda row: sum(abs(row[rank_col] - team_df[rank_col].values[0]) for rank_col in selected_columns if not pd.isna(row[rank_col])),
        axis=1
    )


    similar_teams = matchstats_df[matchstats_df['team_name'] != selected_team]
    similar_teams = similar_teams[similar_teams['team_name']!='Horsens']
    top_3_similar_teams = similar_teams.nsmallest(3, 'similarity_score')
    top_3_similar_teams = top_3_similar_teams.sort_values(by='similarity_score', ascending=False)

    with col2:
        st.write("Teams similar to the selected team:")
        st.dataframe(top_3_similar_teams[['team_name'] + rank_columns + ['similarity_score']], hide_index=True)
    st.header('Central defenders')
    balanced_central_defender_df = balanced_central_defender_df[balanced_central_defender_df['team_name'] == 'Horsens']
    balanced_central_defender_df['match_date'] = pd.to_datetime(balanced_central_defender_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])

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

    # Combine both filters: recent matches and matches involving top teams
    combined_df = pd.merge(recent_matches_df, matches_with_teams_df, how='outer')
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date'])
    # Display the filtered DataFrame
    combined_df = combined_df.drop(columns=['player_position'])

    st.dataframe(combined_df, hide_index=True)
    combined_df = combined_df.drop(columns=['label'])
    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',
        
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}
    }).reset_index()
    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    st.dataframe(agg_df, hide_index=True)
    
    st.header('Fullbacks')
    fullbacks_df = fullbacks_df[fullbacks_df['team_name'] == 'Horsens']
    fullbacks_df.loc[:, 'match_date'] = pd.to_datetime(fullbacks_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    fullbacks_df = fullbacks_df.dropna(subset=['match_date'])
    recent_matches_df = fullbacks_df[fullbacks_df['match_date'].isin(latest_dates)]
    matches_with_teams_df = fullbacks_df[fullbacks_df['label'].str.contains('|'.join(teams_list))]
    combined_df = pd.merge(recent_matches_df, matches_with_teams_df, how='outer')
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date', 'player_position','player_positionSide'])
    st.dataframe(combined_df, hide_index=True)
    combined_df = combined_df.drop(columns=['label'])

    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',
        
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}
    }).reset_index()
    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    st.dataframe(agg_df, hide_index=True)

    st.header('Number 6')
    fullbacks_df = number6_df[number6_df['team_name'] == 'Horsens']
    fullbacks_df.loc[:, 'match_date'] = pd.to_datetime(fullbacks_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    fullbacks_df = fullbacks_df.dropna(subset=['match_date'])
    recent_matches_df = fullbacks_df[fullbacks_df['match_date'].isin(latest_dates)]
    matches_with_teams_df = fullbacks_df[fullbacks_df['label'].str.contains('|'.join(teams_list))]
    combined_df = pd.merge(recent_matches_df, matches_with_teams_df, how='outer')
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date','player_position'])
    st.dataframe(combined_df, hide_index=True)
    combined_df = combined_df.drop(columns=['label'])

    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',
        
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}
    }).reset_index()
    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    st.dataframe(agg_df, hide_index=True)

    st.header('Number 8')
    fullbacks_df = number8_df[number8_df['team_name'] == 'Horsens']
    fullbacks_df.loc[:, 'match_date'] = pd.to_datetime(fullbacks_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    fullbacks_df = fullbacks_df.dropna(subset=['match_date'])
    recent_matches_df = fullbacks_df[fullbacks_df['match_date'].isin(latest_dates)]
    matches_with_teams_df = fullbacks_df[fullbacks_df['label'].str.contains('|'.join(teams_list))]
    combined_df = pd.merge(recent_matches_df, matches_with_teams_df, how='outer')
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date','player_position'])
    st.dataframe(combined_df, hide_index=True)
    combined_df = combined_df.drop(columns=['label'])

    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',
        
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}
    }).reset_index()
    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    st.dataframe(agg_df, hide_index=True)

    st.header('Number 10')
    fullbacks_df = number10_df[number10_df['team_name'] == 'Horsens']
    fullbacks_df.loc[:, 'match_date'] = pd.to_datetime(fullbacks_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    fullbacks_df = fullbacks_df.dropna(subset=['match_date'])
    recent_matches_df = fullbacks_df[fullbacks_df['match_date'].isin(latest_dates)]
    matches_with_teams_df = fullbacks_df[fullbacks_df['label'].str.contains('|'.join(teams_list))]
    combined_df = pd.merge(recent_matches_df, matches_with_teams_df, how='outer')
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date'])
    st.dataframe(combined_df, hide_index=True)
    combined_df = combined_df.drop(columns=['label'])

    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',
        
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}
    }).reset_index()
    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    st.dataframe(agg_df, hide_index=True)

    st.header('Winger')
    fullbacks_df = winger_df[winger_df['team_name'] == 'Horsens']
    fullbacks_df.loc[:, 'match_date'] = pd.to_datetime(fullbacks_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    fullbacks_df = fullbacks_df.dropna(subset=['match_date'])
    recent_matches_df = fullbacks_df[fullbacks_df['match_date'].isin(latest_dates)]
    matches_with_teams_df = fullbacks_df[fullbacks_df['label'].str.contains('|'.join(teams_list))]
    combined_df = pd.merge(recent_matches_df, matches_with_teams_df, how='outer')
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date'])
    st.dataframe(combined_df, hide_index=True)
    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',
        
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}
    }).reset_index()
    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    st.dataframe(agg_df, hide_index=True)

    
    st.header('Striker')
    fullbacks_df = classic_striker_df[classic_striker_df['team_name'] == 'Horsens']
    fullbacks_df.loc[:, 'match_date'] = pd.to_datetime(fullbacks_df['label'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    fullbacks_df = fullbacks_df.dropna(subset=['match_date'])
    recent_matches_df = fullbacks_df[fullbacks_df['match_date'].isin(latest_dates)]
    matches_with_teams_df = fullbacks_df[fullbacks_df['label'].str.contains('|'.join(teams_list))]
    combined_df = pd.merge(recent_matches_df, matches_with_teams_df, how='outer')
    combined_df = combined_df.sort_values(by='Total score', ascending=False)
    combined_df = combined_df.drop(columns=['match_date'])
    st.dataframe(combined_df, hide_index=True)
    combined_df = combined_df.drop(columns=['label'])

    agg_df = combined_df.groupby(['playerName', 'team_name']).agg({
        'minsPlayed': 'sum',
        
        **{col: 'mean' for col in combined_df.columns if col not in ['minsPlayed', 'playerName', 'team_name']}
    }).reset_index()
    agg_df = agg_df.sort_values(by='Total score', ascending=False)
    st.dataframe(agg_df, hide_index=True)
    
def Physical_data():
    df = load_physical_data()
    df.set_index('Team', inplace=True)

    # Preserve the original numeric data for sorting
    numeric_df = df.copy()

    # Format columns to remove thousand separators and ensure proper decimal formatting for display
    formatted_df = df.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else f"{x}")
    st.dataframe(numeric_df)

    # Create ranks for the metrics
    for column in numeric_df.columns:
        rank_col = f"Rank_{column}"
        numeric_df[rank_col] = numeric_df[column].rank(ascending=False).astype(int)

    st.title("Team Performance Metrics")

    # Specify the columns to visualize
    columns_to_plot = [
        'High Speed Running Count',
        'High Speed Running Distance',
        'Sprinting Count',
        'Sprinting Distance',
        'Total Distance'
    ]

    # Create bar charts for each specified column using Plotly
    for column in columns_to_plot:
        
        # Sort the DataFrame by the current column
        sorted_df = numeric_df.sort_values(by=column, ascending=False)
        
        # Plotting
        fig = go.Figure(data=[
            go.Bar(x=sorted_df.index, y=sorted_df[column], marker_color='skyblue')
        ])
        fig.update_layout(
            title=f'{column} - Sorted by Value',
            xaxis_title='Team',
            yaxis_title=column,
            xaxis_tickangle=-45
        )
        
        # Display the plot in Streamlit
        st.plotly_chart(fig)

    # Combine formatted and rank data

    # Display the combined DataFrame with ranks

Data_types = {
    'Dashboard': Dashboard,
    'League stats': League_stats,
    'Physical data': Physical_data
}


st.cache_data(experimental_allow_widgets=True)
st.cache_resource(experimental_allow_widgets=True)
selected_data = st.sidebar.radio('Choose data type',list(Data_types.keys()))

st.cache_data(experimental_allow_widgets=True)
st.cache_resource(experimental_allow_widgets=True)
Data_types[selected_data]()
