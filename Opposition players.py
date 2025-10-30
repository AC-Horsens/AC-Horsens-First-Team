import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
from datetime import datetime
from datetime import date
from matplotlib.path import Path
import unicodedata
team_name = 'Kolding'

def convert_to_ascii(text):
    if isinstance(text, str):
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    return text

def load_data():
    df_xg = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/xg_all DNK_1_Division_2025_2026.csv')
    df_xg['label'] = df_xg['label'] + ' ' + df_xg['date']

    df_xa = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/xA_all DNK_1_Division_2025_2026.csv')
    df_xa['label'] = df_xa['label'] + ' ' + df_xa['date']

    df_pv = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/pv_all DNK_1_Division_2025_2026.csv')
    df_pv['label'] = df_pv['label'] + ' ' + df_pv['date']

    df_possession_stats = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/possession_stats_all DNK_1_Division_2025_2026.csv')
    df_possession_stats['label'] = df_possession_stats['label'] + ' ' + df_possession_stats['date']

    df_xa_agg = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/Horsens/Horsens_possession_data.csv')
    df_xa_agg['label'] = df_xa_agg['label'] + ' ' + df_xa_agg['date']

    df_possession_data = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/Horsens/Horsens_possession_data.csv')
    df_possession_data['label'] = df_possession_data['label'] + ' ' + df_possession_data['date']

    df_xg_agg = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/Horsens/Horsens_xg_data.csv')
    df_xg_agg['label'] = df_xg_agg['label'] + ' ' + df_xg_agg['date']

    df_pv_agg = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/Horsens/Horsens_pv_data.csv')
    df_pv_agg['label'] = df_pv_agg['label'] + ' ' + df_pv_agg['date']

    df_possession_xa = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/xA_all DNK_1_Division_2025_2026.csv')
    df_possession_xa['label'] = df_possession_xa['label'] + ' ' + df_possession_xa['date']

    df_xg_all = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/xg_all DNK_1_Division_2025_2026.csv')
    df_xg_all['label'] = df_xg_all['label'] + ' ' + df_xg_all['date']

    df_pv_all = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/pv_all DNK_1_Division_2025_2026.csv')
    df_pv_all['label'] = df_pv_all['label'] + ' ' + df_pv_all['date']

    df_matchstats = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/matchstats_all DNK_1_Division_2025_2026.csv')
    df_matchstats['label'] = df_matchstats['label'] + ' ' + df_matchstats['date']

    squads = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/squads DNK_1_Division_2025_2026.csv')
    
    possession_events = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/Horsens/Horsens_possession_data.csv')
    possession_events['label'] = possession_events['label'] + ' ' + possession_events['date']
    #packing_df = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/packing_all DNK_1_Division_2025_2026.csv')
    #packing_df['label'] = packing_df['label'] + ' ' + packing_df['date']
    
    #space_control_df = pd.read_csv('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/DNK_1_Division_2025_2026/Space_control_all DNK_1_Division_2025_2026.csv')
    #space_control_df['label'] = space_control_df['label'] + ' ' + space_control_df['date']
    
    return df_xg, df_xa, df_pv, df_possession_stats, df_xa_agg, df_possession_data, df_xg_agg, df_pv_agg, df_xg_all, df_possession_xa, df_pv_all, df_matchstats, squads, possession_events

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
                    4 if row['Passing_'] < 3 else 3,
                    1 if row['Possession_value_added'] < 3 else 1
                ]
            ),
            axis=1
        )
        
        df_balanced_central_defender = df_balanced_central_defender[
            ['playerName', 'team_name', 'player_position', 'minsPlayed','date','label', 'age_today', 'Defending_', 'Possession_value_added', 'Passing_', 'Total score']
        ]
        # Prepare summary
        df_balanced_central_defendertotal = df_balanced_central_defender[
            ['playerName', 'team_name', 'player_position', 'minsPlayed','date', 'age_today', 'Defending_', 'Possession_value_added', 'Passing_', 'Total score']
        ]
        df_balanced_central_defendertotal = df_balanced_central_defendertotal.groupby(['playerName', 'team_name','date', 'player_position', 'age_today']).mean().reset_index()
        df_balanced_central_defendertotal['minsPlayed total'] = df_balanced_central_defender.groupby(
            ['playerName', 'team_name','date', 'player_position', 'age_today'])['minsPlayed'].sum().astype(float).reset_index(drop=True)

        # Filter players with sufficient total minutes and sort
        df_balanced_central_defendertotal = df_balanced_central_defendertotal[
            df_balanced_central_defendertotal['minsPlayed total'].astype(int) >= minutter_total
        ]
        df_balanced_central_defendertotal = df_balanced_central_defendertotal.sort_values('Total score', ascending=False)
        df_balanced_central_defender = df_balanced_central_defender.sort_values('Total score', ascending=False)

        return df_balanced_central_defender

    def fullback():
        mask = (
            (df_scouting['player_position'] == 'Defender') &
            (df_scouting['player_positionSide'].isin(['Right', 'Left'])))
        df_backs = df_scouting[mask].copy()

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
                [3 if row['Defending_'] < 3 else 3, 3 if row['Passing_'] < 2 else 1, 6 if row['Chance_creation'] > 3 else 2, 3 if row['Possession_value_added'] < 3 else 2]
            ), axis=1
        )

        df_backs = df_backs.dropna()

        df_backstotal = df_backs[['playerName', 'team_name','date', 'player_position', 'player_positionSide', 'minsPlayed',
                                'age_today', 'Defending_', 'Passing_', 'Chance_creation', 'Possession_value_added',
                                'Total score']]
        
        df_backs = df_backs[['playerName', 'team_name','date', 'player_position', 'player_positionSide','age_today', 'minsPlayed','label', 'Defending_', 'Passing_', 'Chance_creation','Possession_value_added', 'Total score']]

        df_backstotal = df_backstotal.groupby(['playerName', 'team_name','date', 'player_position', 'player_positionSide', 'age_today']).mean().reset_index()

        minutter = df_backs.groupby(['playerName', 'team_name','date', 'player_position', 'player_positionSide', 'age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_backstotal['minsPlayed total'] = minutter['minsPlayed']

        df_backs = df_backs.sort_values('Total score', ascending=False)
        df_backstotal = df_backstotal[['playerName', 'team_name','date', 'player_position', 'player_positionSide', 'age_today',
                                    'minsPlayed total', 'Defending_', 'Passing_', 'Chance_creation',
                                    'Possession_value_added', 'Total score']]
        df_backstotal = df_backstotal[df_backstotal['minsPlayed total'].astype(int) >= minutter_total]

        df_backstotal = df_backstotal.sort_values('Total score', ascending=False)

        return df_backs
    
    def wingback():
        mask = (
            ((df_scouting['formationUsed'].isin([532, 541])) &
            (df_scouting['player_position'] == 'Defender') &
            (df_scouting['player_positionSide'].isin(['Right', 'Left'])))
            |
            ((df_scouting['formationUsed'].isin([352, 343,3421,3142])) &
            (df_scouting['player_position'] == 'Midfielder') &
            (df_scouting['player_positionSide'].isin(['Right', 'Left'])))
            |
            (df_scouting['player_position'] == 'Wing Back') &
            (df_scouting['player_positionSide'].isin(['Right', 'Left'])))
        
        df_backs = df_scouting[mask].copy()

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
                [3 if row['Defending_'] < 3 else 5, 1 if row['Passing_'] < 2 else 1, 6 if row['Chance_creation'] > 3 else 2, 3 if row['Possession_value_added'] < 3 else 2]
            ), axis=1
        )

        df_backs = df_backs.dropna()

        df_backstotal = df_backs[['playerName', 'team_name','date', 'player_position', 'player_positionSide', 'minsPlayed',
                                'age_today', 'Defending_', 'Passing_', 'Chance_creation', 'Possession_value_added',
                                'Total score']]
        
        df_backs = df_backs[['playerName', 'team_name','date', 'player_position', 'player_positionSide','age_today', 'minsPlayed','label', 'Defending_', 'Passing_', 'Chance_creation','Possession_value_added', 'Total score']]
        
        df_backstotal = df_backstotal.groupby(['playerName', 'team_name','date', 'player_position', 'player_positionSide', 'age_today']).mean().reset_index()

        minutter = df_backs.groupby(['playerName', 'team_name', 'player_position', 'player_positionSide', 'age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_backstotal['minsPlayed total'] = minutter['minsPlayed']

        df_backs = df_backs.sort_values('Total score', ascending=False)
        df_wingbacks = df_backs.copy()

        return df_wingbacks
    
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
            [3 if row['Defending_'] < 5 else 5, 3 if row['Passing_'] < 5 else 4, 3 if row['Progressive_ball_movement'] < 5 else 2, 1 if row['Possession_value_added'] < 5 else 1]
        ), axis=1
        )

        df_sekser = df_sekser[['playerName','team_name','player_position','label','date','minsPlayed','age_today','Defending_','Passing_','Progressive_ball_movement','Possession_value_added','Total score']]
        df_sekser = df_sekser.dropna()
        df_seksertotal = df_sekser[['playerName','team_name','date','player_position','minsPlayed','age_today','Defending_','Passing_','Progressive_ball_movement','Possession_value_added','Total score']]

        df_seksertotal = df_seksertotal.groupby(['playerName','team_name','date','player_position','age_today']).mean().reset_index()
        minutter = df_sekser.groupby(['playerName', 'team_name','date','player_position','age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_seksertotal['minsPlayed total'] = minutter['minsPlayed']
        df_sekser = df_sekser.sort_values('Total score',ascending = False)
        df_seksertotal = df_seksertotal[['playerName','team_name','date','player_position','age_today','minsPlayed total','Defending_','Passing_','Progressive_ball_movement','Possession_value_added','Total score']]
        df_seksertotal= df_seksertotal[df_seksertotal['minsPlayed total'].astype(int) >= minutter_total]
        df_seksertotal = df_seksertotal.sort_values('Total score',ascending = False)

        return df_sekser

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
                1 if row['Progressive_ball_movement'] < 5 else 3, 1 if row['Possession_value'] < 5 else 3]
            ), axis=1
        )

        # Prepare final output
        df_otter = df_otter.dropna()

        df_ottertotal = df_otter[['playerName', 'team_name','date', 'player_position', 'minsPlayed', 'age_today', 
                                'Defending_', 'Passing_', 'Progressive_ball_movement', 'Possession_value', 'Total score']]
        
        df_otter = df_otter[['playerName', 'team_name','date', 'player_position', 'age_today', 'minsPlayed', 'label', 
                            'Defending_', 'Passing_', 'Progressive_ball_movement', 'Possession_value', 'Total score']]

        df_ottertotal = df_ottertotal.groupby(['playerName', 'team_name','date', 'player_position', 'age_today']).mean().reset_index()
        minutter = df_otter.groupby(['playerName', 'team_name','date', 'player_position', 'age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_ottertotal['minsPlayed total'] = minutter['minsPlayed']

        df_otter = df_otter.sort_values('Total score', ascending=False)
        df_ottertotal = df_ottertotal[['playerName', 'team_name','date', 'player_position', 'age_today', 'minsPlayed total', 
                                    'Defending_', 'Passing_', 'Progressive_ball_movement', 'Possession_value', 'Total score']]
        df_ottertotal = df_ottertotal[df_ottertotal['minsPlayed total'].astype(int) >= minutter_total]

        df_ottertotal = df_ottertotal.sort_values('Total score', ascending=False)

        return df_otter

    def number10():
        mask = (
            (
                (df_scouting['formationUsed'].isin([343, 3421, 541, 4231, 4321])) &
                (df_scouting['player_position'].isin(['Attacking Midfielder', 'Striker'])) &
                (df_scouting['player_positionSide'].isin(['Centre/Right', 'Left/Centre']))
            )
            |
            (
                (df_scouting['player_position'] == 'Attacking Midfielder') &
                (df_scouting['player_positionSide'].isin(['Centre','Centre/Right','Left/Centre']))
            )
        )

        df_10 = df_scouting[mask].copy()

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

        df_10total = df_10[['playerName', 'team_name','date', 'minsPlayed', 'age_today', 
                            'Passing_', 'Chance_creation', 'Goalscoring_', 'Possession_value', 'Total score']]
        
        df_10 = df_10[['playerName', 'team_name','date', 'age_today', 'minsPlayed', 'label', 
                    'Passing_', 'Chance_creation', 'Goalscoring_', 'Possession_value', 'Total score']]

        df_10total = df_10total.groupby(['playerName', 'team_name','date', 'age_today']).mean().reset_index()
        minutter = df_10.groupby(['playerName', 'team_name','date', 'age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_10total['minsPlayed total'] = minutter['minsPlayed']

        df_10 = df_10.sort_values('Total score', ascending=False)
        df_10total = df_10total[['playerName', 'team_name','date', 'age_today', 'minsPlayed total', 
                                'Passing_', 'Chance_creation', 'Goalscoring_', 'Possession_value', 'Total score']]
        df_10total = df_10total[df_10total['minsPlayed total'].astype(int) >= minutter_total]

        df_10total = df_10total.sort_values('Total score', ascending=False)

        return df_10

    def winger():
        mask = (
            ((df_scouting['formationUsed'].isin([442,541,451,4141])) &
            (df_scouting['player_position'] == 'Midfielder') &
            (df_scouting['player_positionSide'].isin(['Right', 'Left'])))
            |
            ((df_scouting['formationUsed'].isin([433])) &
            (df_scouting['player_position'] == 'Striker') &
            (df_scouting['player_positionSide'].isin(['Left/Centre', 'Centre/Right'])))
            |
            (df_scouting['player_position'].isin(['Attacking Midfielder', 'Striker'])) &
            (df_scouting['player_positionSide'].isin(['Right', 'Left'])))        

        df_winger = df_scouting[mask].copy()

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
        
        df_winger = df_winger[['playerName', 'team_name','date', 'age_today', 'minsPlayed', 'label', 
                    'Passing_', 'Chance_creation', 'Goalscoring_', 'Possession_value', 'Total score']]

        df_winger_total = df_winger[['playerName', 'team_name','date', 'minsPlayed', 
                                    'age_today', 'Passing_', 'Chance_creation', 'Goalscoring_', 'Possession_value', 'Total score']]
        df_winger_total = df_winger_total.groupby(['playerName', 'team_name','date', 'age_today']).mean().reset_index()
        minutter = df_winger.groupby(['playerName', 'team_name','date', 'age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_winger_total['minsPlayed total'] = minutter['minsPlayed']

        df_winger_total = df_winger_total[df_winger_total['minsPlayed total'].astype(int) >= minutter_total]
        df_winger_total = df_winger_total.sort_values('Total score', ascending=False)
        df_winger = df_winger.sort_values('Total score', ascending=False)

        return df_winger

    def Classic_striker():
        mask = (
        ((df_scouting['formationUsed'].isin([532,442,352,3142,3412])) &
        (df_scouting['player_position'] == 'Striker') &
        (df_scouting['player_positionSide'].str.contains('Centre')))
        |
        (df_scouting['player_position'] == 'Striker') &
        (df_scouting['player_positionSide'] == 'Centre'))

        df_striker = df_scouting[mask].copy()
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
        df_striker = df_striker.dropna()
        df_striker= df_striker[['playerName', 'team_name','date', 'age_today', 'minsPlayed', 'label', 
                    'Linkup play', 'Chance creation', 'Goalscoring', 'Possession value', 'Total score']]

        df_striker_total = df_striker[['playerName', 'team_name','date', 'minsPlayed', 
                                    'age_today', 'Linkup play', 'Chance creation', 'Goalscoring', 'Possession value', 'Total score']]
        df_striker_total = df_striker_total.groupby(['playerName', 'team_name','date', 'age_today']).mean().reset_index()
        minutter = df_striker.groupby(['playerName', 'team_name','date', 'age_today'])['minsPlayed'].sum().astype(float).reset_index()
        df_striker_total['minsPlayed total'] = minutter['minsPlayed']

        df_striker_total = df_striker_total[df_striker_total['minsPlayed total'].astype(int) >= minutter_total]
        df_striker_total = df_striker_total.sort_values('Total score', ascending=False)
        df_striker = df_striker.sort_values('Total score',ascending = False)
        return df_striker

    return {
        'Central defender': balanced_central_defender(),
        'Fullback':fullback(),
        'Wingback': wingback(),
        'Number 6' : number6(),
        'Number 8': number8(),
        'Number 10': number10(),
        'Winger': winger(),
        'Striker': Classic_striker(),
    }

df_xg, df_xa, df_pv, df_possession_stats, df_xa_agg, df_possession_data, df_xg_agg, df_pv_agg, df_xg_all, df_possession_xa, df_pv_all, df_matchstats, squads, possession_events = load_data()
df_horsens_seneste = df_xg[df_xg['team_name'] == team_name]
dates = df_horsens_seneste['date'].drop_duplicates().sort_values()
dates = dates[-5:]
position_dataframes = Process_data_spillere(df_possession_xa, df_pv, df_matchstats, df_xg_all, squads)

#defending_central_defender_df = position_dataframes['defending_central_defender']
#ball_playing_central_defender_df = position_dataframes['ball_playing_central_defender']
balanced_central_defender_df = position_dataframes['Central defender']
fullbacks_df = position_dataframes['Fullback']
wingbacks_df = position_dataframes['Wingback']
number6_df = position_dataframes['Number 6']
#number6_double_6_forward_df = position_dataframes['number6_double_6_forward']
#number6_destroyer_df = position_dataframes['Number 6 (destroyer)']
number8_df = position_dataframes['Number 8']
number10_df = position_dataframes['Number 10']
winger_df = position_dataframes['Winger']
classic_striker_df = position_dataframes['Striker']
#targetman_df = position_dataframes['Targetman']
#box_striker_df = position_dataframes['Boxstriker']
#horsens_df, merged_df, total_expected_points_combined = process_data()
def create_pdf_progress_report_4_matches(position_dataframes):
    MIN_MINUTES = 0  # threshold for season part

    today = date.today()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Add the team logo
    pdf.image('C:/Users/Seamus-admin/Documents/GitHub/AC-Horsens-First-Team/Logo.png', 
              x=165, y=5, w=10, h=10)
    pdf.set_xy(5, 5)
    pdf.cell(25, 5, txt=f"Progress report: {today}", ln=True, align='L')

    # --- Horsens tables (full season, with 300 min filter) ---
    y_position = 30
    pdf.set_xy(5, y_position)
    for position, df in position_dataframes.items():
        dfx = df.copy()
        dfx['Total score'] = pd.to_numeric(dfx['Total score'], errors='coerce')
        dfx = dfx.dropna(subset=['Total score'])

        # Eligible players by minutes
        if 'minsPlayed' in dfx.columns:
            mins_total = dfx.groupby('playerName', as_index=False)['minsPlayed'].sum()
            eligible = set(mins_total.loc[mins_total['minsPlayed'] >= MIN_MINUTES, 'playerName'])
            dfx_eligible = dfx[dfx['playerName'].isin(eligible)]
        else:
            eligible = None
            dfx_eligible = dfx

        # Horsens subset
        horsens = dfx[dfx['team_name'] == team_name].copy()
        horsens = horsens[horsens['date'].isin(dates)]
        print(horsens)
        if horsens.empty:
            continue
        if eligible is not None:
            horsens = horsens[horsens['playerName'].isin(eligible)]

        # Drop unused columns
        drop_cols = ['label', 'team_name', 'age_today', 
                     'player_position', 'player_positionSide']
        horsens.drop(columns=[c for c in drop_cols if c in horsens.columns], 
                     inplace=True, errors='ignore')

        # Aggregate per player
        numeric_columns = horsens.select_dtypes(include='number').columns.tolist()
        if 'minsPlayed' in numeric_columns:
            numeric_columns.remove('minsPlayed')
        aggregation_dict = {col: 'mean' for col in numeric_columns}
        if 'minsPlayed' in horsens.columns:
            aggregation_dict['minsPlayed'] = 'sum'

        table = horsens.groupby('playerName').agg(aggregation_dict).reset_index()
        if 'minsPlayed' in table.columns:
            table = table[table['minsPlayed'] >= MIN_MINUTES]

        # Reorder and sort
        reordered_columns = ['playerName']
        if 'minsPlayed' in table.columns:
            reordered_columns += ['minsPlayed']
        reordered_columns += [c for c in numeric_columns if c in table.columns]
        table = table[reordered_columns].round(2)
        table = table.sort_values('Total score', ascending=False)

        # Render
        pdf.set_font("Arial", size=6)
        pdf.cell(190, 4, 
                 txt=convert_to_ascii(f"Position Report: {position} (Ranked, {team_name}, Season)"), 
                 ln=True, align='C')
        headers = table.columns.tolist()
        col_widths = [min(max(len(h)*2.2, 15), 35) for h in headers]
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 4, txt=convert_to_ascii(h), border=1)
        pdf.ln(4)

        for _, row in table.iterrows():
            total_score = row['Total score']
            fill_color = (255,0,0) if total_score < 4 else (255,255,0) if total_score <= 6 else (0,255,0)
            pdf.set_fill_color(*fill_color)
            for i, val in enumerate(row):
                pdf.cell(col_widths[i], 4, txt=convert_to_ascii(str(val)), border=1, fill=True)
            pdf.ln(4)


    pdf.output(f"Progress reports/Progress_report_{team_name}_{today}.pdf")
    print(f'{today} progress report created')

create_pdf_progress_report_4_matches(position_dataframes)


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
