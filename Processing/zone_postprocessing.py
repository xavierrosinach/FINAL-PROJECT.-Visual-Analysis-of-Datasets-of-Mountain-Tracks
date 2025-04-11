import pandas as pd
import os
import json
import numpy as np
from datetime import datetime
import requests
from geopy.distance import geodesic
import warnings
import shutil
from zone_preprocessing import zone_preprocessing

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

bounds_dict = {"canigo": (2.2, 2.7, 42.4, 42.6),
               "matagalls": (2.3, 2.5, 41.8, 41.9),
               "vallferrera": (1.2, 1.7, 42.5, 42.8)}

center_coords_dict = {"canigo": (2.5, 42.5), 
                      "matagalls": (2.4, 41.825), 
                      "vallferrera": (1.35, 42.6)}


# Function to print the message and append it in the log file
def log(message, log_file=None):
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

# Checks eges that have been visited only once in a neiborhood, and concatenates the previous edge id
def clean_single_edges(merged_fmm_out):
    ids = merged_fmm_out['id'].tolist()

    for i in range(1, len(ids) - 1):
        prev_id = ids[i - 1]
        curr_id = ids[i]
        next_id = ids[i + 1]

        # Si el id actual es diferente de anterior y siguiente,
        # y anterior == siguiente, entonces está "aislado"
        if curr_id != prev_id and curr_id != next_id and prev_id == next_id:
            ids[i] = prev_id  # Reemplazar con el id del entorno

    merged_fmm_out['id'] = ids
    return merged_fmm_out

# Converts the string date into a correct date
def convert_date(date):

    # Replace types of appostrofes
    date = date.replace("de gener", "January")
    date = date.replace("de febrer", "February")
    date = date.replace("de març", "March")
    date = date.replace("d’abril", "April")
    date = date.replace("d'abril", "April")
    date = date.replace("de maig", "May")
    date = date.replace("de juny", "June")
    date = date.replace("de juliol", "July")
    date = date.replace("d’agost", "August")
    date = date.replace("d'agost", "August")
    date = date.replace("de setembre", "September")
    date = date.replace("d’octubre", "October")
    date = date.replace("d'octubre", "October")
    date = date.replace("de novembre", "November")
    date = date.replace("de desembre", "December")

    # Remove the extra "de"
    date = date.replace(" de ", " ")

    # Convert to datetime
    date = datetime.strptime(date.strip(), "%d %B %Y")

    return date.strftime("%d/%m/%Y")

# Cleans the coordinates df and returns other parametes
def clean_coordinates_df(coords_df, fmm_out, edges):

    coords = coords_df[['lon','lat','elev']].copy()     # Select the needed columns

    # First, create shifted lat/lon columns to compare with previous row
    coords['prev_lat'] = coords['lat'].shift()
    coords['prev_lon'] = coords['lon'].shift()

    # Calculate geodesic distance between current and previous point
    coords['dist'] = coords.apply(lambda row: geodesic((row['prev_lat'], row['prev_lon']), (row['lat'], row['lon'])).meters
                                if pd.notnull(row['prev_lat']) else 0,
                                axis=1)

    # Optional: drop the helper columns if you don't need them
    coords.drop(['prev_lat', 'prev_lon'], axis=1, inplace=True)

    # Sum the distance as a cumulative sumatory
    coords['dist'] = round(coords['dist'].cumsum() / 1000, 2)

    # For each edge matched, get the id
    merged_fmm_out = fmm_out.merge(edges, on=['u', 'v'])    # Merge to obtain the id
    merged_fmm_out = merged_fmm_out[['lon', 'lat', 'id']]       # Select some columns
    merged_fmm_out = clean_single_edges(merged_fmm_out)     # Clean the single edge-ids with the created function
    merged_fmm_out = merged_fmm_out.rename(columns = {'lon':'clean_lon', 'lat':'clean_lat', 'id':'edge_id'})    # Rename columns for a better understanding

    # Concatenate the two dataframes to obtain the final one
    cleaned_coords = pd.concat([coords, merged_fmm_out], axis=1)

    # Add an index for each coordinate and the actual km
    cleaned_coords['id'] = range(1, len(cleaned_coords) + 1)
    cleaned_coords['km'] = np.floor(cleaned_coords['dist']).astype(int)

    # Calculate other statistics
    total_dist = cleaned_coords['dist'].iloc[-1]
    cleaned_coords['elev_diff'] = cleaned_coords["elev"].diff()
    elev_gain = round(cleaned_coords[cleaned_coords["elev_diff"] > 0]["elev_diff"].sum(), 2)
    elev_loss = round(abs(cleaned_coords[cleaned_coords["elev_diff"] < 0]["elev_diff"].sum()), 2) 

    # First and last coordinates
    first_lat = cleaned_coords['lat'].iloc[0]
    first_lon = cleaned_coords['lon'].iloc[0]
    last_lat = cleaned_coords['lat'].iloc[-1]
    last_lon = cleaned_coords['lon'].iloc[-1]

    return cleaned_coords[['id','edge_id','lon','lat','elev','dist','km','clean_lon','clean_lat']], total_dist, elev_gain, elev_loss, first_lat, first_lon, last_lat, last_lon

# Function to obtain the weather information of a given zone from one date to another
def obtain_weather_dataframe(start_date, end_date, zone):

    # Transform the start and the end dates
    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    # Obtain the center cords
    center_cords = center_coords_dict[zone]
    lon = center_cords[0]
    lat = center_cords[1]

    # Convert weather code to a readable condition
    weather_mapping = {0: "Clear", 1: "Clear",          # The library has more weather conditions, but we define less groups
                       2: "Cloudy", 3: "Cloudy",
                       45: "Fog", 48: "Fog",
                       51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
                       61: "Rain", 63: "Rain", 65: "Rain", 80: "Rain", 81: "Rain", 82: "Rain",
                       71: "Snow", 73: "Snow", 75: "Snow",
                       95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm"}
    
    url = (f"https://archive-api.open-meteo.com/v1/archive?"    # Url with the information
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_min,temperature_2m_max,weathercode"
            f"&timezone=auto")

    data = requests.get(url).json()     # The request to json format

    df = pd.DataFrame({"date": data["daily"]["time"],   # Result to dataframe
                        "min_temp": data["daily"]["temperature_2m_min"],
                        "max_temp": data["daily"]["temperature_2m_max"],
                        "weather_code": data["daily"]["weathercode"]})
    
    df['weather_condition'] = df['weather_code'].map(weather_mapping)

    return df[['date', 'min_temp', 'max_temp', 'weather_condition']]

# Function to determine season based on exact date
def get_season(date):
    month = date.month
    day = date.day

    if (month == 12 and day >= 22) or (month in [1, 2]) or (month == 3 and day <= 21):
        return 'Winter'
    elif (month == 3 and day >= 22) or (month in [4, 5]) or (month == 6 and day <= 21):
        return 'Spring'
    elif (month == 6 and day >= 22) or (month in [7, 8]) or (month == 9 and day <= 21):
        return 'Summer'
    elif (month == 9 and day >= 22) or (month in [10, 11]) or (month == 12 and day <= 21):
        return 'Autumn'
    
# Cleans the output dataframe with the final desired format
def final_output_cleaning(df, zone):

    # Dates processing
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.strftime('%B')
    df['weekday'] = df['date'].dt.day_name()  # Gives full weekday name (e.g., 'Monday')
    df['season'] = df['date'].apply(get_season)      # Apply the function

    # Obtain the weather dataframe
    weather_df = obtain_weather_dataframe(df['date'].min(), df['date'].max(), zone)

    # Prepare df to merge it with weather df
    df = df.drop(columns=['min_temp','max_temp','weather_condition'])
    df['date'] = pd.to_datetime(df['date'])
    weather_df['date'] = pd.to_datetime(weather_df['date'])

    merged_df = df.merge(weather_df, on='date', how='inner')    # Merge df with weather df

    # Select only needed columns
    final_df = merged_df[['track_id','url','user','difficulty','distance','elev_gain','elev_loss','date','year','month','weekday','season',
                          'min_temp','max_temp','weather_condition','first_lat','first_lon','last_lat','last_lon']]     

    return final_df

# Full post-processing function for the given zone
def zone_postprocessing(zone, log_path):

    # Obtain all paths and dataframes
    input_data_path, osm_data_path, output_path, dataframes_path, fmm_out_path, cleaned_tracks_path, disc_df, disc_df_path, out_df, out_df_path, edges_df_path, edges_df, clean_out_df_path, clean_out_df, waypoints_df_path, waypoints_df, logs_path = zone_preprocessing(zone)

    total_files = len(out_df)

    list_cleaned_tracks = clean_out_df['track_id'].unique().tolist()

    log(f'Starting the POSTPROCESSING PATH of "{zone}". Cleaning each track.')

    for index, row in out_df.iterrows():

        # Obtaint the track id and inform
        track_id = int(row['track_id'])  

        if track_id not in list_cleaned_tracks:

            log(f'  Processing track {track_id}. {index+1} of {total_files}.', log_path)

            input_file = os.path.join(input_data_path, str(track_id) + '.json')      # Input file (JSON)
            out_fmm = pd.read_csv(os.path.join(fmm_out_path, str(track_id) + '.csv'))       # FMM output file

            # Load JSON data from the file
            with open(input_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Get the desired information
            url = data.get("url")
            user = int(data.get("user"))
            date = convert_date(data.get("date-up"))
            difficulty = data.get("difficulty")
            coords, total_dist, elev_gain, elev_loss, first_lat, first_lon, last_lat, last_lon = clean_coordinates_df(pd.DataFrame(data["coordinates"], columns=["lon", "lat", "elev", "unkn"]), out_fmm, edges_df)

            # Filter activities - we do not want outliers
            if (total_dist >= 3) & (total_dist < 30) & (elev_gain < 5000) & (elev_loss < 5000):

                # Add the information into the cleaned dataframe
                clean_out_df = pd.concat([clean_out_df, pd.DataFrame({'track_id':[track_id], 'url':[url], 'user':[user], 'difficulty':[difficulty], 'distance':[total_dist],
                                                                    'elev_gain':[elev_gain], 'elev_loss':[elev_loss], 'date':[date], 'first_lat':[first_lat], 'first_lon':[first_lon],
                                                                    'last_lat':[last_lat], 'last_lon':[last_lon]})], ignore_index=True)
                clean_out_df = clean_out_df.drop_duplicates(subset=['track_id'])      # Avoid duplications
                clean_out_df.to_csv(clean_out_df_path, index=False)     # Save the dataframe on each iteration to avoid missing data

                # Save the coords dataframe
                coords.to_csv(os.path.join(cleaned_tracks_path, str(track_id) + '.csv'), index=False)

            else:       # Update the discarded files dataframe
                disc_df = pd.concat([disc_df, pd.DataFrame({'track_id':[track_id], 'zone':[zone], 'error_type':[6]})], ignore_index=True)
                disc_df.to_csv(disc_df_path, index=False)   # Concatenate the info with the discard dataframe and save it
        
    # Obtain the final output dataframe and save it
    final_out_df = final_output_cleaning(clean_out_df, zone).reset_index()
    final_out_df.to_csv(clean_out_df_path, index=False)

    # Process the edges
    total_files = len(final_out_df)

    log(f'Processing the edges.')
    for index, row in final_out_df.iterrows():

        track_id = row['track_id']  # Obtain the track id
        log(f'  Processing edges for file {track_id}. {index+1} of {total_files}.' , log_path)

        clean_df = pd.read_csv(os.path.join(cleaned_tracks_path, str(track_id)+'.csv'))
        list_edges = clean_df['edge_id'].unique().tolist()

        # Actualize the edges list
        for edge in list_edges:
            mask = (edges_df['id'] == int(edge))
            current_list = edges_df.loc[mask, 'list_tracks'].values[0]

            # Only update count_tracks and list_tracks if track_id is not already in the list
            if str(track_id) not in current_list:
                edges_df.loc[mask, 'total_tracks'] += 1          # Update count_tracks
                edges_df.loc[mask, 'list_tracks'] = edges_df.loc[mask, 'list_tracks'].apply(lambda lst: lst + [str(track_id)] if isinstance(lst, list) else [str(track_id)])

    # Weight the edges total tracks column
    edges_df = edges_df[edges_df['total_tracks'] > 0]
    edges_df['weight'] = round(1 + ((edges_df['total_tracks'] - edges_df['total_tracks'].min()) / (edges_df['total_tracks'].max() - edges_df['total_tracks'].min())) * 9).astype(int)

    empty_edges = edges_df[edges_df['total_tracks'] == 0]       # Empty edges (avoid missing edges)
    edges_df = pd.concat([edges_df, empty_edges], ignore_index=True)        # Concatenate the dataframes

    # Save the edges df
    edges_df.to_csv(edges_df_path, index=False)

    log(f"Postprocessing done for the zone '{zone}'.", log_path)
    log('----------', log_path)