import pandas as pd
import os
import json
import geopandas as gpd
import ast  # To safely evaluate string representations of lists
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
import requests
import warnings

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

bounds_dict = {"canigo": (2.2, 2.7, 42.4, 42.6), "matagalls": (2.3, 2.5, 41.8, 41.9), "vallferrera": (1.2, 1.7, 42.5, 42.8)}
center_coords_dict = {"canigo": (2.5, 42.5), "matagalls": (2.4, 41.825), "vallferrera": (1.35, 42.6)}

# Function to obtain the weather information of a given zone from one date to another
def obtain_weather_dataframe(start_date, end_date, lat, lon):

    # Convert weather code to a readable condition
    weather_mapping = {0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog",
                       48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
                       61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 71: "Slight snow", 73: "Moderate snow",
                       75: "Heavy snow", 80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
                       95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"}
    
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

def coordinates_cleaning(coord_df, edges_df):
    
    coord_df = coord_df[['Longitude', 'Latitude', 'Elevation', 'MatchedLatitude', 'MatchedLongitude', 'source', 'target']].copy()      # Select desired columns
    coord_df[['source', 'target']] = coord_df.apply(lambda row: sorted([row['source'], row['target']]), axis=1, result_type='expand')   # Apply order to source and target
    coord_df = coord_df.merge(edges_df, left_on=['source', 'target'], right_on=['u', 'v'], how='inner')     # Merge dataframes
    coord_df['Elevation-Diff'] = coord_df["Elevation"].diff()
    coord_df = coord_df[['Latitude','Longitude','Elevation','Elevation-Diff','edge_id']]    # Only desired columns

    return coord_df

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
        return 'Fall'
    
def output_df_postproc(zone, output_df):

        # Create a copy of the output dataframe - selecting only desired columns
    postproc_output_df = output_df[['track_id','url','title','user','date_up','activity_type_name','difficulty','distance']].copy()

    # Main filter of the tracks - only 'Senderisme' activities
    postproc_output_df = postproc_output_df[postproc_output_df['activity_type_name'] == 'Senderisme']

    # Column changes
    postproc_output_df['distance'] = round(postproc_output_df['distance']/1000, 2)   # Distance to km
    postproc_output_df['difficulty'] = postproc_output_df['difficulty'].map({'Fàcil':1, 'Moderat':2, 'Difícil':3, 'Molt difícil':4, 'Només experts':5})     # Convert the difficulty into a string

    # Catalan to English month mapping
    month_mapping = {"de gener":"January", "de febrer":"February", "de març":"March", "d’abril":"April",
                    "de maig":"May", "de juny":"June", "de juliol":"July", "d’agost":"August",
                    "de setembre":"September", "d’octubre":"October", "de novembre":"November", "de desembre":"December"}

    # Convert Catalan dates to English
    for ca, en in month_mapping.items():
        postproc_output_df['date_up'] = postproc_output_df['date_up'].str.replace(ca, en, regex=True)
    postproc_output_df['date_up'] = pd.to_datetime(postproc_output_df['date_up'], format="%d %B de %Y").dt.strftime("%Y-%m-%d")     # Convert to datetime

    # Center coordinates
    center_coords = center_coords_dict[zone]

    # Obtain the weather dataframe
    weather_df = obtain_weather_dataframe(postproc_output_df['date_up'].min(), postproc_output_df['date_up'].max(), center_coords[1], center_coords[0])
    postproc_output_df = postproc_output_df.merge(weather_df, left_on='date_up', right_on='date')

    # Add an elevation gain and loss columns
    postproc_output_df['elevation_gain'] = 0.0
    postproc_output_df['elevation_loss'] = 0.0
    
    # First and last coordinates initialization
    postproc_output_df['first_lat'] = 0.0
    postproc_output_df['first_lon'] = 0.0
    postproc_output_df['last_lat'] = 0.0
    postproc_output_df['last_lon'] = 0.0
    postproc_output_df['geometry'] = postproc_output_df.apply(lambda _: [], axis=1)     # Empty list

    # Dates processing
    postproc_output_df['date_up'] = pd.to_datetime(postproc_output_df['date_up'])
    postproc_output_df['year'] = postproc_output_df['date_up'].dt.year
    postproc_output_df['month'] = postproc_output_df['date_up'].dt.strftime('%B')
    postproc_output_df['weekday'] = postproc_output_df['date_up'].dt.day_name()  # Gives full weekday name (e.g., 'Monday')
    postproc_output_df['season'] = postproc_output_df['date_up'].apply(get_season)      # Apply the function

    return postproc_output_df

# Function to extract the first URL from the 'photo' column
def extract_first_url(photo_list):
    if isinstance(photo_list, str):  # Convert string to list if necessary
        try:
            photo_list = ast.literal_eval(photo_list)
        except:
            return None  # Return None if conversion fails
    if isinstance(photo_list, list) and len(photo_list) > 0:
        return photo_list[0].get('url', None)  # Extract 'url' if exists
    return None

def waypoints_cleaning(zone, track_id, waypoints_df):

    # Define the bounds and filter the waypoints only inside
    bounds = bounds_dict[zone]
    waypoints_df = waypoints_df[(waypoints_df['lon']>=bounds[0]) & (waypoints_df['lon']<=bounds[1]) & (waypoints_df['lat']>=bounds[2]) & (waypoints_df['lat']<=bounds[3])]
    waypoints_df['url'] = waypoints_df['photos'].apply(extract_first_url)   # Obtain only the FIRST url of a photo (if it has one)

    waypoints_df['track_id'] = track_id
    waypoints_df = waypoints_df[['id','track_id','lat','lon','elevation','pictogramName','url']]   # Select desired columns
    waypoints_df = waypoints_df.rename(columns={'id':'waypoint_id','lat':'latitude','lon':'longitude','pictogramName':'type'})

    return waypoints_df

def track_postproc(zone, track_id, edges_df, tracks_path):

    track_path = os.path.join(tracks_path, str(track_id) + '.json')

    # Load JSON data from the file
    with open(track_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Read the coordinates and waypoints df
    coord_df = pd.DataFrame(data.get("coordinates"))
    wayp_df = pd.DataFrame(data.get("waypoints"))

    # Coordinates cleaning
    coord_df = coordinates_cleaning(coord_df, edges_df)

    # Waypoints cleaning
    if not wayp_df.empty:
        wayp_df = waypoints_cleaning(zone, track_id, wayp_df)

    # Obtain the list of edges that the track goes by
    list_edges = coord_df['edge_id'].unique().tolist()

    return coord_df, wayp_df, list_edges

def update_edges_df(zone, edges_df, list_tracks, post_proc_path, all_waypoints_df, tracks_path, output_df):

    for track_id in list_tracks:

        coord_df, wayp_df, list_edges = track_postproc(zone, track_id, edges_df, tracks_path)

        # Add elevation loss and gain for each track
        output_df.loc[output_df['track_id'] == track_id, 'elevation_gain'] = round(coord_df[coord_df["Elevation-Diff"] > 0]["Elevation-Diff"].sum(), 2)
        output_df.loc[output_df['track_id'] == track_id, 'elevation_loss'] = round(abs(coord_df[coord_df["Elevation-Diff"] < 0]["Elevation-Diff"].sum()), 2) 

        # First and last coordinates
        output_df.loc[output_df['track_id'] == track_id, 'first_lat'] = coord_df['Latitude'].iloc[0]
        output_df.loc[output_df['track_id'] == track_id, 'first_lon'] = coord_df['Longitude'].iloc[0]
        output_df.loc[output_df['track_id'] == track_id, 'last_lat'] = coord_df['Latitude'].iloc[-1]
        output_df.loc[output_df['track_id'] == track_id, 'last_lon'] = coord_df['Longitude'].iloc[-1]

        coord_df.to_csv(os.path.join(post_proc_path, str(track_id)+'.csv'), index=False)

        # Upgrade the waypoints df
        all_waypoints_df = pd.concat([all_waypoints_df, wayp_df], ignore_index=True)

        # Actualize the edges list
        for edge in list_edges:
            mask = (edges_df['edge_id'] == edge)
            current_list = edges_df.loc[mask, 'list_tracks'].values[0]

            # Only update count_tracks and list_tracks if track_id is not already in the list
            if track_id not in current_list:
                edges_df.loc[mask, 'count_tracks'] += 1          # Update count_tracks
                edges_df.loc[mask, 'list_tracks'] = edges_df.loc[mask, 'list_tracks'].apply(lambda lst: lst + [track_id])           # Append track_id to list_tracks

    return edges_df, all_waypoints_df, output_df

def coloring_tracks(list_tracks, heat_edges_df, post_proc_path, output_df):

    color_dict = {1: '#FFFF00',  # Yellow
                  2: '#FFD700',  # Gold
                  3: '#FFCC00',  # Yellow-Orange
                  4: '#FF8000',  # Orange
                  5: '#FF4500',  # Orange-Red
                  6: '#FF0000',  # Red
                  7: '#B22222',  # Firebrick
                  8: '#8B0000',  # Dark Red
                  9: '#800000',  # Maroon
                  10: '#600000'} # Dark Maroon

    for track_id in list_tracks:

        coord_df = pd.read_csv(os.path.join(post_proc_path, str(track_id)+'.csv'))
        coord_df = coord_df.merge(heat_edges_df, left_on=['edge_id'], right_on=['edge_id'], how='inner')     # Merge dataframes
        coord_df['color'] = coord_df['heat_edge'].apply(lambda x: color_dict.get(x, '#FFFFFF'))  # Default to white if not found
        coord_df = coord_df.rename(columns={'Latitude':'lat','Longitude':'lon','Elevation':'elev','edge_id':'edge','color':'edge_color','heat_edge':'edge_weight'})
        coord_df = coord_df[['lat','lon','elev','edge','edge_color','edge_weight']]
        coord_df.to_csv(os.path.join(post_proc_path, str(track_id)+'.csv'), index=False)

        # Add the geometry (list of [lat, lon, edge_color])
        # output_df.loc[output_df['track_id'] == track_id, 'geometry'] = coord_df[['lat', 'lon', 'edge_color']].values.tolist()
        output_df.at[output_df.index[output_df['track_id'] == track_id][0], 'geometry'] = coord_df[['lat', 'lon', 'edge_color']].values.tolist()


    return output_df

def postprocessing_zone(zone):
    
    # Obtain all the paths
    data_path = '../../Data'
    osm_data = os.path.join(data_path, 'OSM-Data', zone)
    output_path = os.path.join(data_path, 'Output', zone)
    tracks_path = os.path.join(output_path, 'tracks')

    # Read dataframes
    output_df = pd.read_csv(os.path.join(output_path, 'output_files.csv'))
    output_df = output_df_postproc(zone, output_df)

    # Read edges shapefile as df - and clean it
    edges_df = gpd.read_file(os.path.join(osm_data, "edges.shp"))
    edges_df = edges_df[['u', 'v', 'geometry']]     # Select desired columns
    edges_df[['u', 'v']] = edges_df.apply(lambda row: sorted([row['u'], row['v']]), axis=1, result_type='expand')   # Apply order to source and target
    edges_df = edges_df.drop_duplicates(subset=['u', 'v'], keep='first')    # Avoid duplicated rows
    edges_df['edge_id'] = range(1, len(edges_df) + 1)       # Apply an id column to identify edges
    edges_df['count_tracks'] = 0        # Column to add +1 if a track goes through it
    edges_df['list_tracks'] = [[] for _ in range(len(edges_df))]    # List of tracks that goes through that edge

    # Path to add all the post processed tracks
    post_proc_path = os.path.join(output_path, 'postproc_tracks')
    os.makedirs(post_proc_path, exist_ok=True)

    # Create an empty dataframe for the waypoints
    all_waypoints_df = pd.DataFrame(columns=['waypoint_id', 'track_id', 'latitude', 'longitude', 'elevation', 'type', 'url'])

    list_tracks = output_df['track_id'].unique().tolist()   # List of the tracks for the post-processing

    # Update the edges dataframe
    edges_df, all_waypoints_df, output_df = update_edges_df(zone, edges_df, list_tracks, post_proc_path, all_waypoints_df, tracks_path, output_df)

    # Edges df cleaning and normalizing
    edges_df = edges_df[edges_df['count_tracks'] > 0]       # Select edges that have been passed by at least once   
    edges_df['heat_edge'] = ((1 + (edges_df['count_tracks'] - edges_df['count_tracks'].min()) * 9 / (edges_df['count_tracks'].max() - edges_df['count_tracks'].min()))).round().astype(int) # Normalize the counting from 1 to 10

    # Select only the edge id and their heat value to add it to each coordinates df
    heat_edges_df = edges_df[['edge_id','heat_edge']].copy()

    # Color all the edges of the tracks
    output_df = coloring_tracks(list_tracks, heat_edges_df, post_proc_path, output_df)

    # Reorder columns
    output_df = output_df[['track_id','url','title','user','difficulty','distance','elevation_gain','elevation_loss','date','year','month','weekday',
                           'season','min_temp','max_temp','weather_condition','first_lat','first_lon','last_lat','last_lon','geometry']]

    # Save all dataframes
    all_waypoints_df.to_csv(os.path.join(output_path, 'all_waypoints.csv'), index=False)
    output_df.to_csv(os.path.join(output_path, 'cleaned_output.csv'), index=False)
    edges_df.to_csv(os.path.join(output_path, 'cleaned_edges.csv'), index=False)

postprocessing_zone('matagalls')