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
print('connected to database')
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

from concurrent.futures import ThreadPoolExecutor
import psycopg2
import pandas as pd
import io
from psycopg2 import sql

def fast_table_to_df(cur, schema, table):
    query = sql.SQL('COPY {}.{} TO STDOUT WITH CSV HEADER').format(
        sql.Identifier(schema),
        sql.Identifier(table)
    )
    buffer = io.StringIO()
    cur.copy_expert(query, buffer)
    buffer.seek(0)
    return pd.read_csv(buffer)

def process_schema(schema):
    local_conn = psycopg2.connect(
        dbname=db_navn,
        user=db_brugernavn,
        password=db_adgangskode,
        host=db_host,
        port="5432"
    )
    local_cur = local_conn.cursor()

    result_dfs = []
    try:
        local_cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (schema,))
        tables = [t[0] for t in local_cur.fetchall() if 'possession_data' in t[0]]

        for table in tables:
            print(f"Processing {schema}.{table}")
            df = fast_table_to_df(local_cur, schema, table)
            df = map_to_unified_columns_possession(df)
            df = df[['timeMin', 'timeSec', 'x', 'y','team_name', 'playerName', 'label','date','typeId','sequence_duration','sequenceId','sequence_xG','5.0', '6.0', '9.0', '24.0', '25.0', '26.0', '107.0']]
            result_dfs.append(df)
    except Exception as e:
        print(f"Error processing {schema}: {e}")
    finally:
        local_cur.close()
        local_conn.close()

    return result_dfs

schemas_to_include = [
    #'DNK_1_Division_2023_2024',
    'DNK_1_Division_2025_2026',
    #'DNK_Superliga_2023_2024',
    #'DNK_Superliga_2024_2025'
]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_schema, schemas_to_include))

# Flatten and combine
all_data_frames = [df for dfs in results for df in dfs]
combined_df = pd.concat(all_data_frames, ignore_index=True)



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

def filter_out_possession_ids_fast(df):
    columns_to_check = ['5.0', '6.0', '9.0', '24.0', '25.0', '26.0', '107.0']

    # Ensure the relevant columns are strings
    df[columns_to_check] = df[columns_to_check].astype(str).apply(lambda col: col.str.lower())

    # Create a mask of rows where any of the columns are 'true'
    mask = (df[columns_to_check] == 'true').any(axis=1)

    # Get a set of tuples representing combinations to exclude
    exclude_set = set(zip(df.loc[mask, 'date'], df.loc[mask, 'label'], df.loc[mask, 'sequenceId']))

    # Use .isin with a tuple comparison for efficient filtering
    keep_mask = ~df[['date', 'label', 'sequenceId']].apply(tuple, axis=1).isin(exclude_set)

    return df[keep_mask]
print('df created')
# Apply filter function
df = filter_out_possession_ids_fast(combined_df)
print('data filtered')

def add_start_possession_distance(df):
    # Find the rows where possession_index == 1 (start of possession)
    start_distance = df[df['possession_index'] == 1][['label', 'date', 'sequenceId', 'distance to opp goal']].copy()
    
    # Rename the column to indicate that it’s the starting possession distance
    start_distance.rename(columns={'distance to opp goal': 'start_possession_distance'}, inplace=True)
    
    # Merge this start distance back into the original DataFrame, based on sequenceId, label, and date
    df = df.merge(start_distance, on=['label', 'date', 'sequenceId'], how='left')
    
    return df

def add_start_possession_distance_efficient(df):
    # Sorter først for korrekt rækkefølge
    df = df.sort_values(by=['label', 'date', 'sequenceId', 'timeMin', 'timeSec'], ignore_index=True)

    # Brug .groupby().first() i stedet for .loc og filter
    start_distance = (
        df[df['possession_index'] == 1]
        .set_index(['label', 'date', 'sequenceId'])['distance to opp goal']
        .rename('start_possession_distance')
    )

    # Brug join i stedet for merge
    df = df.join(start_distance, on=['label', 'date', 'sequenceId'])

    return df

    # Ensure x and y are numeric

def length_to_opp_goal(df):
    # Ensure x and y are numeric
    df.loc[:, 'x'] = pd.to_numeric(df['x'], errors='coerce')
    df.loc[:, 'y'] = pd.to_numeric(df['y'], errors='coerce')

    # Calculate distance to opponent's goal
    goal_x = 100
    goal_y = 50
    df.loc[:, 'distance to opp goal'] = np.sqrt((goal_x - df['x'])**2 + (goal_y - df['y'])**2)
    
    return df

df = length_to_opp_goal(df)

print('length to opponnents goal defined')

df = df.sort_values(by=['label', 'date', 'timeMin', 'timeSec'])  # eller tilsvarende kolonner
df['possession_index'] = df.groupby(['label','date','sequenceId']).cumcount() + 1

# Add start possession distance
df = add_start_possession_distance_efficient(df)
print('start possession distance added')
# Convert '321.0' to float

# Merge this new 'possession_xg' back into the original DataFrame
df['sequence_duration'] = df['sequence_duration'].fillna(0).astype(int)

# Apply conditions for filtering
condition1 = (df['start_possession_distance'].astype(float) <= 30) & (df['sequence_duration'].astype(int) <= 5)
condition2 = (df['start_possession_distance'].astype(float) >= 30) & (df['start_possession_distance'].astype(float) <= 60) & (df['sequence_duration'].astype(int) <= 8)
condition3 = (df['start_possession_distance'].astype(float) >= 60) & (df['sequence_duration'].astype(int) <= 11)

# Combine all conditions with OR (|)
df1 = df[condition1 | condition2 | condition3]
# Filter based on team name and possession_xg > 0
print('Conditions applied')
# Select relevant columns
#df1 = df1[['timeMin', 'timeSec', 'x', 'y','team_name', 'playerName', 'label', 'distance to opp goal', 'start_possession_distance', 'possession_length', 'possession_index', 'possession_xg']]
long_possessions = (
    df.groupby(['label', 'date', 'sequenceId'])['possession_index']
    .max()
    .reset_index()
)

# Behold kun dem hvor der er mere end 2 hændelser i possessionen
long_possessions = long_possessions[long_possessions['possession_index'] > 2]

# Merge for at filtrere df1 baseret på lange possessions
df1 = df1.merge(long_possessions[['label', 'date', 'sequenceId']], 
                on=['label', 'date', 'sequenceId'], 
                how='inner')
df1 = df1.sort_values(by=['label', 'date', 'timeMin', 'timeSec'])  # eller tilsvarende kolonner

# Print the final DataFrame
df1 = df1[df1['sequence_xG'] > 0.1]


from mplsoccer import Pitch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
df_start = df1[df1['possession_index'] == 1]
def plot_heatmap_location(data):
    pitch = Pitch(pitch_type='opta', line_zorder=2, pitch_color='grass', line_color='white')
    fig, ax = pitch.draw(figsize=(6.6, 4.125))
    fig.set_facecolor('#22312b')
    bin_statistic = pitch.bin_statistic(data['x'], data['y'], statistic='count', bins=(50, 25))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot')
    plt.show(fig)

#plot_heatmap_location(df_start)
#df_xgc_transitions = df1.groupby(['playerName','team_name']).sum('sequence_xG').reset_index()
#df_xgc_transitions = df_xgc_transitions.sort_values(by=['sequence_xG'],ascending=False)  # eller tilsvarende kolonner
df1.to_csv(r' Transitions DNK_1_Division_2025_2026.csv')