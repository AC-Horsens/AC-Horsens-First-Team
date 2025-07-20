import http.client
import os
import json
import time
import requests
import pandas as pd
from pandas import json_normalize
from io import StringIO
import json
import re
import numpy as np
from io import StringIO


# Define constants
CLIENT_ID = "LvOY8iNxB9YkeRYa7i1ICRo0yV32k1ac"
CLIENT_SECRET = "lyiBFIhw5cVe6P7XYs8U812uQDazZHiU7mhYFltluReBWzhyjV_eDb7XZF-I4pvc"
HOME_DIR = os.getenv("HOME") or os.getenv("HOMEPATH") or os.getenv("USERPROFILE")
TOKEN_CACHE_DIR = os.getenv('SSI_TOKEN_CACHE') or f"."
AUTH_DOMAIN = "secondspectrum.auth0.com"
HEADERS = {'content-type': "application/x-www-form-urlencoded"}

outlet_auth_key = "18pwnl3lmtgcs14bks8ns28epx"
tournament_calendar_uuid = '351g9crny6x5a4uxcc90s2904'

db_navn = 'AC Horsens'
db_brugernavn = 'postgres'
db_adgangskode = 'achorsens'
db_host = 'localhost'
db_schema = 'DNK_1_Division_2024_2025'
# Define the function to get the access token
def get(audience):
    cacheKey = getCacheKey(CLIENT_ID, audience)
    tokenFile = f"{TOKEN_CACHE_DIR}/{cacheKey}.json"
    fsToken = getFSToken(tokenFile)
    if fsToken != None and fsToken['token'] != None and time.time() <= int(fsToken['expires']):
        return fsToken['token']

    newToken = fetchTokenClientCreds(
        CLIENT_ID, CLIENT_SECRET, audience, AUTH_DOMAIN)

    with open(tokenFile, "w+") as f:
        json.dump(newToken, f)

    return newToken['token']

def getCacheKey(id, audienceName):
    return f"{AUTH_DOMAIN}_{id}_{audienceName}"

def getFSToken(tokenFile):
    if os.path.exists(tokenFile):
        with open(tokenFile, "r") as f:
            tokenStr = f.readline()
            token = json.loads(tokenStr)
            return token

def fetchTokenClientCreds(clientID, clientSecret, audience, authDomain):
    conn = http.client.HTTPSConnection(authDomain)
    payload = f"grant_type=client_credentials&client_id={clientID}&client_secret={clientSecret}&audience={audience}"
    conn.request("POST", "/oauth/token", payload, HEADERS)

    res = conn.getresponse().read().decode("utf-8")
    data = json.loads(res)

    tokenData = dict()
    tokenData['token'] = data['access_token']
    tokenData['expires'] = int(time.time()) + int(data['expires_in'])
    return tokenData

def get_schedule():
    # Define the API endpoint URL
    api_url = "https://api.secondspectrum.com"
    # Define the audience for the API
    audience = "hermes-api-external.prod"
    # Define the competition ID and season
    competition_id = "f31c7af4-17bd-420d-81e1-21590dbc8eed"
    year = "2024"
    # Get the access token using the get function
    access_token = get(audience)
    # Define the headers for the HTTP request with the access token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    # Define the endpoint to fetch the schedule
    endpoint = f"/schedule/ssi?competitionId={competition_id}&season={year}"
    # Make the HTTP GET request to the API endpoint
    response = requests.get(api_url + endpoint, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Process the data as needed
        if 'data' in data:
            game_data = data['data']
            df = pd.DataFrame(game_data)
            return df
        else:
            print("No game data found in the response.")
            return None
    else:
        # Print an error message if the request failed
        print(f"Error: {response.status_code} - {response.text}")
        return None

def get_metadata(game_id):
    # Define the API endpoint URL
    api_url = "https://api.secondspectrum.com"
    # Define the audience for the API
    audience = "hermes-api-external.prod"
    # Get the access token using the get function
    access_token = get(audience)
    # Define the headers for the HTTP request with the access token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    # Define the endpoint to fetch the game list
    endpoint = f"/gamedata/ssi/meta.json?gameId={game_id}"
    # Make the HTTP GET request to the API endpoint
    response = requests.get(api_url + endpoint, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON Lines response
        data = response.json()
        lines = response.text.splitlines()
        return data
    else:
        # Print an error message if the request failed
        print(f"Error: {response.status_code} - {response.text}")
        return None, None

def get_tracking_produced(game_id):
    # Define the API endpoint URL
    api_url = "https://api.secondspectrum.com"
    # Define the audience for the API
    audience = "hermes-api-external.prod"
    # Get the access token using the get function
    access_token = get(audience)
    # Define the headers for the HTTP request with the access token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    # Define the endpoint to fetch the tracking-produced data
    endpoint = f"/gamedata/ssi/tracking-produced.jsonl?gameId={game_id}"
    
    # Make the HTTP GET request to the API endpoint
    response = requests.get(api_url + endpoint, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Return the JSON Lines response
        return response.text
    else:
        # Print an error message if the request failed
        print(f"Error: {response.status_code} - {response.text}")
        return None

def get_physical_produced(game_id):
    # Define the API endpoint URL
    api_url = "https://api.secondspectrum.com"
    # Define the audience for the API
    audience = "hermes-api-external.prod"
    # Get the access token using the get function
    access_token = get(audience)
    # Define the headers for the HTTP request with the access token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    # Define the endpoint to fetch the tracking-produced data
    endpoint = f"/gamedata/ssi/physical-splits.csv?gameId={game_id}"
    
    # Make the HTTP GET request to the API endpoint
    response = requests.get(api_url + endpoint, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Return the JSON Lines response
        return response.text
    else:
        # Print an error message if the request failed
        print(f"Error: {response.status_code} - {response.text}")
        return None

def get_physical_summary(game_id):
    # Define the API endpoint URL
    api_url = "https://api.secondspectrum.com"
    # Define the audience for the API
    audience = "hermes-api-external.prod"
    # Get the access token using the get function
    access_token = get(audience)
    # Define the headers for the HTTP request with the access token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    # Define the endpoint to fetch the tracking-produced data
    endpoint = f"/gamedata/ssi/physical-summary.csv?gameId={game_id}"
    
    # Make the HTTP GET request to the API endpoint
    response = requests.get(api_url + endpoint, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Return the JSON Lines response
        return response.text
    else:
        # Print an error message if the request failed
        print(f"Error: {response.status_code} - {response.text}")
        return None

def extract_players(physical_data):
    pattern = r'\"([^\"]+) \((\d{6})\)\"'
    matches = re.findall(pattern, physical_data)
    player_list = [(match[0], match[1]) for match in matches]
    return player_list

def extract_metrics(physical_data, potential_teams):
    columns = ['Metric', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '50.1', '55', '60', '65', '70', '75', '80', '85', '90', '95', '100']
    lines = physical_data.strip().split('\n')
    metrics_data = []
    current_player = None
    
    for line in lines:
        # Check for player names
        name_match = re.match(r'^"([^"]+) \((\d+)\)"$', line)
        if name_match:
            current_player = name_match.group(1)
            continue
        
        # Skip team names
        if current_player in potential_teams:
            continue
        
        # Check for metric rows
        metric_match = re.match(r'^"([^"]+)",(.+)$', line)
        if metric_match:
            metric_name = metric_match.group(1)
            values = metric_match.group(2).split(',')
            if len(values) < len(columns) - 1:
                values += [''] * (len(columns) - 1 - len(values))
            metrics_data.append([current_player, metric_name] + values)
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    return df

def calculate_average_excluding_zeros(row,numeric_columns):
    # Extract numeric values
    values = row[numeric_columns].replace(0, np.nan)  # Replace zeros with NaN
    # Calculate average, ignoring NaNs
    return values.mean()

def convert_minutes_to_timedelta(minutes_str):
    try:
        # Convert the mm:ss format to a pandas timedelta
        minutes, seconds = map(int, minutes_str.split(':'))
        return pd.to_timedelta(minutes, unit='m') + pd.to_timedelta(seconds, unit='s')
    except ValueError:
        # If conversion fails, return NaT (Not a Time)
        return pd.NaT



potential_teams = [
    "Boldklubben af 1893", "Kolding IF", "Esbjerg fB", "Odense Boldklub", "FC Fredericia", "Vendsyssel FF", 
    "Hillerod Fodbold", "Hobro IK", "HB Koge", "Hvidovre IF", "AC Horsens", "FC Roskilde"
]

all_match_data = pd.DataFrame(columns=['Team', 'Metric', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '50.1', '55', '60', '65', '70', '75', '80', '85', '90', '95', '100', 'Match', 'Sum'])
all_player_data = pd.DataFrame()
all_matches_summarized = pd.DataFrame()
schedule_df = get_schedule()
schedule_df = schedule_df[schedule_df['description'].str.contains('EFB')]
if schedule_df is not None:
    for index, row in schedule_df.iterrows():
            game_id = row['ssiId']
            try:

                metadata = get_metadata(game_id)
                metadata_df = pd.json_normalize(metadata)
                description = metadata_df['description'].iloc[0]  # Extract the first element
                
                # Clean the description to make it a valid filename
                description = re.sub(r'[\\/*?:"<>|]', "_", description).strip()

                print(metadata_df[['ssiId', 'description']])

                # Get the physical data
                physical_data = get_physical_produced(game_id)
                physical_summary = get_physical_summary(game_id)
                physical_summary = StringIO(physical_summary)

                physical_summary = pd.read_csv(physical_summary, skiprows=9)
                # Select the desired columns
                physical_summary_df = physical_summary[['ID', 'Player', 'Minutes', 'Distance', 'High Speed Running', 'Sprinting', 'No. of High Intensity Runs', 'Top Speed']]
                physical_summary_df['Distance'] = pd.to_numeric(physical_summary_df['Distance'], errors='coerce')
                physical_summary_df = physical_summary_df[physical_summary_df['Distance'].notna()]
                physical_summary_df['Top Speed'] = pd.to_numeric(physical_summary_df['Top Speed'], errors='coerce')

                # Apply the conversion function to 'Minutes'
                physical_summary_df.dropna(subset=['Minutes'], inplace=True)

                # Calculate average top speed
                all_matches_summarized = pd.concat([all_matches_summarized, physical_summary_df], ignore_index=True)
            except Exception as e:
                print(f"Error processing game ID {game_id}: {e}")

# Save all matches data to a single CSV file
all_matches_summarized = all_matches_summarized.sort_values(by=['Player', 'Top Speed'], ascending=[True, False]) \
                .groupby('Player') \
                .apply(lambda x: x.iloc[2:]) \
                .reset_index(drop=True)
all_matches_summarized = all_matches_summarized.groupby(['Player', 'ID'], as_index=False).max()
all_matches_summarized = all_matches_summarized.round(2)
all_matches_summarized.to_csv('all_matches_summarized.csv', index=False)

