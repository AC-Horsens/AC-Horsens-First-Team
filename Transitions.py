import pandas as pd
import numpy as np
import psycopg2

db_navn = 'AC Horsens'
db_brugernavn = 'postgres'
db_adgangskode = 'ACHorsens'
db_host = 'localhost'
conn = psycopg2.connect(
    dbname=db_navn,
    user=db_brugernavn,
    password=db_adgangskode,
    host=db_host
)

cur = conn.cursor()
cur.execute("SELECT schema_name FROM information_schema.schemata")
schemas = cur.fetchall()

def map_to_unified_columns_possession(df):
    # Create a dictionary mapping the original column names to the unified column names
    column_mapping = {
        'contestantId': 'contestantId',
        'team_name': 'team_name',
        'id': 'id',
        'eventId': 'eventId',
        'typeId': 'typeId',
        'periodId': 'periodId',
        'timeMin': 'timeMin',
        'timeSec': 'timeSec',
        'outcome': 'outcome',
        'x': 'x',
        'y': 'y',
        'timeStamp': 'timeStamp',
        'lastModified': 'lastModified',
        'playerId': 'playerId',
        'playerName': 'playerName',
        'sequenceId': 'sequenceId',
        'possessionId': 'possessionId',
        'keyPass': 'keyPass',
        'assist': 'assist',
        '140.0': '140.0',
        '141.0': '141.0',
        '318.0': '318.0',
        '321.0': '321.0',
        '210.0': '210',
        '22.0': '22.0',
        '23.0': '23.0',
        '5.0' : '5.0',
        '6.0' : '6.0',
        '9.0' : '9.0',
        '24.0': '24.0',
        '25.0': '25.0',
        '26.0': '26.0',
        '107.0':'107.0',
        '210.0':'210.0',
        '213.0':'213.0',
        'match_id': 'match_id',
        'label': 'label',
        'date': 'date',
    }
    for col in column_mapping.values():
        if col not in df.columns:
            df[col] = None

    # Rename columns using the mapping dictionary
    unified_df = df.rename(columns=column_mapping)

    return unified_df

def fetch_and_save_possession_data_for_nordic_bet_teams():
    try:
        conn = psycopg2.connect(
            dbname=db_navn,
            user=db_brugernavn,
            password=db_adgangskode,
            host=db_host,
            port="5432"
        )
        cur = conn.cursor()

        cur.execute('SELECT table_name FROM information_schema.tables WHERE table_schema = %s', ('DNK_1_Division_2024_2025',))
        tables = cur.fetchall()

        relevant_tables = [table[0] for table in tables if table[0] and 'possession_data' in table[0]]

        all_data_frames = []
        for table_name in relevant_tables:
            cur.execute(f'SELECT * FROM "DNK_1_Division_2024_2025"."{table_name}"')
            data = cur.fetchall()
            column_names = [desc[0] for desc in cur.description]

            # Create a DataFrame with the fetched data
            df = pd.DataFrame(data, columns=column_names)

            # Map data to the correct columns in the unified DataFrame
            unified_df = map_to_unified_columns_possession(df)

            all_data_frames.append(unified_df)

        # Concatenate all DataFrames into one
        combined_df = pd.concat(all_data_frames, ignore_index=True)
    
    except Exception as e:
        print(f"Error fetching and saving data for : {e}")
    finally:
        if 'cur' in locals() and cur is not None:
            cur.close()
        if 'conn' in locals() and conn is not None:
            conn.close()
    return combined_df

df = fetch_and_save_possession_data_for_nordic_bet_teams()


# Reset index to avoid ambiguity with 'label'
df = df.reset_index(drop=True)

def filter_out_possession_ids(df):
    # List of columns to check for 'true' values as strings
    columns_to_check = ['5.0', '6.0', '9.0', '24.0', '25.0', '26.0', '107.0']
    
    # Convert columns to string explicitly
    df[columns_to_check] = df[columns_to_check].astype(str)
    
    # Check if any of the columns contain 'true'
    filter_condition = df[columns_to_check].apply(lambda x: x == 'true').any(axis=1)
    
    # Get the unique combinations of 'date', 'label', and 'sequenceId' that should be excluded
    exclusion_combinations = df.loc[filter_condition, ['date', 'label', 'sequenceId']].drop_duplicates()

    # Filter the DataFrame to exclude rows where 'date', 'label', and 'sequenceId' match any of the exclusion combinations
    filtered_data = df.merge(exclusion_combinations, on=['date', 'label', 'sequenceId'], how='left', indicator=True)
    
    # Only keep rows that are not part of the exclusion combinations
    filtered_data = filtered_data[filtered_data['_merge'] == 'left_only'].drop('_merge', axis=1)
    
    return filtered_data

# Apply filter function
df = filter_out_possession_ids(df)

def calculate_possession_length(df):
    # Ensure timeMin and timeSec are numeric
    df['timeMin'] = pd.to_numeric(df['timeMin'], errors='coerce')
    df['timeSec'] = pd.to_numeric(df['timeSec'], errors='coerce')

    # Group by sequenceId to get the time ranges for each possession
    possession_length = df.groupby(['label','date','sequenceId']).apply(lambda group: (
        (group['timeMin'].max() * 60 + group['timeSec'].max()) - 
        (group['timeMin'].min() * 60 + group['timeSec'].min())
    ))
    # Rename the series for clarity
    possession_length.name = 'possession_length'
    
    return possession_length

sequence_length = calculate_possession_length(df)

def add_start_possession_distance(df):
    # Find the rows where possession_index == 1 (start of possession)
    start_distance = df[df['possession_index'] == 1][['label', 'date', 'sequenceId', 'distance to opp goal']].copy()
    
    # Rename the column to indicate that it’s the starting possession distance
    start_distance.rename(columns={'distance to opp goal': 'start_possession_distance'}, inplace=True)
    
    # Merge this start distance back into the original DataFrame, based on sequenceId, label, and date
    df = df.merge(start_distance, on=['label', 'date', 'sequenceId'], how='left')
    
    return df

def length_to_opp_goal(df):
    # Ensure x and y are numeric
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # Calculate distance to opponent's goal
    goal_x = 100
    goal_y = 50
    df['distance to opp goal'] = np.sqrt((goal_x - df['x'])**2 + (goal_y - df['y'])**2)
    return df

df = length_to_opp_goal(df)

# Merge possession length
df = df.merge(sequence_length, on=['label','date','sequenceId'], how='left')

# Calculate possession index
df['possession_index'] = df.groupby(['label','date','sequenceId']).cumcount() + 1

# Add start possession distance
df = add_start_possession_distance(df)

# Convert '321.0' to float
df['321.0'] = pd.to_numeric(df['321.0'], errors='coerce')

# Calculate possession xG
Transition_xg = df.groupby(['label','date','sequenceId'])['321.0'].max().reset_index()

# Rename the '321.0' column to 'Transition_xg' for clarity
Transition_xg.rename(columns={'321.0': 'Transition xG'}, inplace=True)

# Merge this new 'Transition_xg' back into the original DataFrame
df = df.merge(Transition_xg, on=['label','date','sequenceId'], how='left')

# Apply conditions for filtering
condition1 = (df['start_possession_distance'].astype(float) <= 30) & (df['possession_length'].astype(int) <= 5)
condition2 = (df['start_possession_distance'].astype(float) >= 30) & (df['start_possession_distance'].astype(float) <= 60) & (df['possession_length'].astype(int) <= 8)
condition3 = (df['start_possession_distance'].astype(float) >= 60) & (df['possession_length'].astype(int) <= 11)

# Combine all conditions with OR (|)
df1 = df[condition1 | condition2 | condition3]
# Filter based on team name and Transition_xg > 0

# Select relevant columns
#df1 = df1[['timeMin', 'timeSec', 'x', 'y','team_name', 'playerName', 'label', 'distance to opp goal', 'start_possession_distance', 'possession_length', 'possession_index', 'Transition_xg']]

# Print the final DataFrame
#df1 = df1[df1['Transition_xg'] > 0]
df1 = df1[df1['possession_index'] == 1]
df1['label'] = (df1['label'] + ' ' + df1['date'])

df_by_team = df1.groupby(['team_name','label'])['Transition xG'].sum().reset_index()
df_by_team['match_Transition_xG'] = df_by_team.groupby(['label'])['Transition xG'].transform('sum')
df_by_team['Transition xG difference'] = df_by_team['Transition xG'] - df_by_team['match_Transition_xG'] + df_by_team['Transition xG']
df_by_team['Transition xG against'] = df_by_team['Transition xG'] - df_by_team['match_Transition_xG']
difference_df = df_by_team.groupby(['team_name','label'])['Transition xG difference'].sum().reset_index()
xg_against = df_by_team.groupby(['team_name','label'])['Transition xG against'].sum().reset_index()
total_df = df_by_team.groupby(['team_name','label'])['Transition xG'].sum().reset_index()
combined_df = difference_df.merge(total_df, on=['team_name','label'],how='outer')
combined_df = combined_df.merge(xg_against, on=['team_name','label'],how='outer')
combined_df = combined_df.sort_values('Transition xG difference', ascending=False)
combined_df = combined_df.round(2)

df_xg = pd.read_csv(r'C:\Users\Seamus-admin\Documents\GitHub\AC-Horsens-First-Team\DNK_1_Division_2024_2025\xg_all DNK_1_Division_2024_2025.csv')
df_xg['label'] = (df_xg['label'] + ' ' + df_xg['date'])

df_xg['321'] = df_xg['321'].astype(float)
df_xg = df_xg.groupby(['team_name','label']).sum('321')
df_xg = df_xg.rename(columns={'321': 'Total xG'})
combined_df = combined_df.merge(df_xg,on=['team_name','label'],how='outer')
combined_df['Transition xG share'] = combined_df['Transition xG']/combined_df['Total xG']
combined_df = combined_df[['team_name','label','Transition xG difference','Transition xG','Transition xG against','Transition xG share']]
combined_df['team_name'] = combined_df['team_name'].str.replace(' ', '_')
combined_df = combined_df.set_index('team_name')
combined_df = combined_df.round(2)

combined_df.to_csv(r'C:\Users\Seamus-admin\Documents\GitHub\AC-Horsens-First-Team\DNK_1_Division_2024_2025\Transitions_all DNK_1_Division_2024_2025.csv')