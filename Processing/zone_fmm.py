import pandas as pd
import os
import ast
import json
from shapely.wkt import loads
import signal
import sys
from shapely.geometry import LineString
from geopy.distance import geodesic
from fmm import Network,NetworkGraph,UBODTGenAlgorithm,UBODT,FastMapMatch, FastMapMatchConfig
from zone_preprocessing import zone_preprocessing
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

data_path = '../../Data'
bounds_dict = {"canigo": (2.2, 2.7, 42.4, 42.6), 
               "matagalls": (2.3, 2.5, 41.8, 41.9), 
               "vallferrera": (1.2, 1.7, 42.5, 42.8)}

class TimeoutException(Exception):     # Timeout exception creation
    pass

def timeout_handler(signum, frame):     # Function to break if the timeout is reached
    raise TimeoutException("Function execution timed out")

# Function to print the message and append it in the log file
def log(message, log_file=None):
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

# Function to check if we need to discard the file depending on the coordinates df
def discard_coordinates(zone, coordinates_df, min_coordinates=100, max_distance=300, min_total_distance=1000):

    bounds = bounds_dict.get(zone)  # Define the bounds using the dictionary
    total_distance = 0.0    # Initialize a value for the total distance
   
    # Obtain the total sum of coordinates that are inside the defined bounds
    inside_bounds = ((coordinates_df["Longitude"] >= bounds[0]).all() &
                     (coordinates_df["Longitude"] <= bounds[1]).all() &
                     (coordinates_df["Latitude"] >= bounds[2]).all() &
                     (coordinates_df["Latitude"] <= bounds[3]).all())
    
    # Check if all coordinates are inside the bound
    if not inside_bounds:
        return False, 1

    # Discard the track if it has less than min_coordinates
    if len(coordinates_df) < min_coordinates:
        return False, 2 

    # Check the distance between two consecutive coordinates
    for i in range(len(coordinates_df) - 1):  # Stop at the second last row 
        lon1, lat1 = coordinates_df.loc[i, 'Longitude'], coordinates_df.loc[i, 'Latitude']
        lon2, lat2 = coordinates_df.loc[i + 1, 'Longitude'], coordinates_df.loc[i + 1, 'Latitude']
        
        distance = geodesic((lat1, lon1), (lat2, lon2)).meters          # Compute distance (in meters) using geopy
        total_distance += distance              # Sum the distance at the total distance

        if distance > max_distance:             # Check if the distance between two points is less than the maximum defined
            return False, 3

    # Check if the total distance is greater than the minimum
    if total_distance < min_total_distance:
        return False, 4

    return True, 0      # If any error, return true, and we can proceed with the track processing

# Extract the information of the json file as a dictionary - waypoints and coordinates as dataframes
def extract_information(json_path):

    # Load JSON data from the file
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Obtain the two dataframes
    activity_type = data.get("activity", {}).get("name")
    coords_df = pd.DataFrame(data["coordinates"], columns=["Longitude", "Latitude", "Elevation", "Unknown"])
    wayp_df = pd.DataFrame(data.get("waypoints", []))

    return activity_type, coords_df, wayp_df

# Function to extract the first URL from the 'photo' column of waypoints dataframe
def extract_first_url(photo_list):
    if isinstance(photo_list, str):  # Convert string to list if necessary
        try:
            photo_list = ast.literal_eval(photo_list)
        except:
            return None  # Return None if conversion fails
    if isinstance(photo_list, list) and len(photo_list) > 0:
        return photo_list[0].get('url', None)  # Extract 'url' if exists
    return None

# Function to clean the waypoints dataframe
def waypoints_df_cleaning(waypoints_df, track_id, zone):

    if waypoints_df.empty:   # Check if the dataframe is empty
        return
    
    try:
        waypoints = waypoints_df[['id','lat','lon','elevation','pictogramName','photos']].copy()      # Select some columns
        waypoints = waypoints.rename(columns={'elevation':'elev', 'pictogramName':'type'})      # Rename some columns
        waypoints['url'] = waypoints['photos'].apply(extract_first_url)   # Obtain only the FIRST url of a photo (if it has one)
        waypoints = waypoints.drop(columns=['photos'])      # Drop the column photos
        waypoints = waypoints.rename(columns={'id':'waypoint_id'})      # Rename the column
        waypoints['track_id'] = track_id        # Add the track as id

        # Select only the waypoints that are inside the defined bounds
        bounds = bounds_dict[zone]
        waypoints = waypoints[(waypoints['lon'] >= bounds[0]) & (waypoints['lon'] <= bounds[1]) &
                              (waypoints['lat'] >= bounds[2]) & (waypoints['lat'] <= bounds[3])]

        return waypoints
    
    except:
        return

# Function to obtain the matching track - training process
def matching_track(model, coordinates_df, radius, gps_error, timeout=60):

    k_candidates = [2, 3, 4]        # Define the k candidates

    # Get the track wkt
    line = LineString(zip(coordinates_df["Longitude"], coordinates_df["Latitude"]))
    track_wkt = line.wkt

    # Try for all candidates
    for k in k_candidates:

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)  # Trigger alarm after `timeout` seconds

        try:    # Try to obtain a matching track with the defined configuration
            config = FastMapMatchConfig(k, radius, gps_error)   
            result = model.match_wkt(track_wkt, config)
            signal.alarm(0)  # Reset alarm

            if result.pgeom.export_wkt().strip() != "LINESTRING()":     # If a result is found, return it
                return True, 0, result, k, radius, gps_error

        except TimeoutException:    # If the timeout of 60 seconds is reached
            continue

        finally:    # When we finish, we reset the clock
            signal.alarm(0)
    
    return False, 5, None, 0, 0, 0  # Return error if any matching track is found

# Function to save the output file
def save_fmm_output(file_output_path, fmm_result):
    
    new_track = fmm_result.pgeom.export_wkt()   # Export the fmm output
    line = loads(new_track)     # Convert to shapely LineString object
    coords = list(line.coords)      # Extract coordinates

    df = pd.DataFrame(columns=["lon", "lat"])   # Create empty dataframe
    df["lon"], df["lat"] = zip(*coords)     # Insert the coordinates

    # Obtain the information of the candidates
    candidates = []
    for c in fmm_result.candidates:
        candidates.append((c.source,c.target))
    edges = pd.DataFrame(candidates, columns=["u","v"])

    # Concatenate dataframes
    df = pd.concat([df.reset_index(drop=True), edges.reset_index(drop=True)], axis=1)
    df[['u', 'v']] = df.apply(lambda row: sorted([row['u'], row['v']]), axis=1, result_type='expand')   # Apply order

    # Save the dataframe
    df.to_csv(file_output_path, index=False)

# Main function
def zone_fmm(zone, log_path):

    # Obtain all dataframes and paths
    input_data_path, osm_data_path, output_path, dataframes_path, fmm_out_path, cleaned_tracks_path, disc_df, disc_df_path, out_df, out_df_path, edges_df_path, edges_df, clean_out_df_path, clean_out_df, waypoints_df_path, waypoints_df, logs_path = zone_preprocessing(zone)
    log('----------', log_path)
    log(f'Starting the PROCESSING PART of "{zone}" (Fast Map Matching).', log_path)
    log('----------', log_path)


    # Obtain the FastMapMatching model of the zone
    network = Network(os.path.join(osm_data_path, 'edges.shp'), "fid", "u", "v")     # Network of the zone with FMM
    graph = NetworkGraph(network)   # Network graph of the zone with FMM

    # Create or read the UDOBT file
    udobt_path = os.path.join(osm_data_path, 'udobt.txt')
    if not os.path.exists(udobt_path):      # If it is not created
        ubodt_gen = UBODTGenAlgorithm(network,graph)
        status = ubodt_gen.generate_ubodt(udobt_path, 4, binary=False, use_omp=True)
    ubodt = UBODT.read_ubodt_csv(udobt_path)    # If it is created, read it

    model = FastMapMatch(network,graph,ubodt)   # Creation of the model using FMM

    # Get the list of the output files and discarded files of the zone
    disc_files = disc_df['track_id'].unique().tolist()
    out_files = out_df['track_id'].unique().tolist()
    processed_files = list(set(disc_files + out_files))

    for track in os.listdir(input_data_path):

        track_id = int(track.split('.')[0])           # Obtain the track id

        if track_id not in processed_files:
            track_path = os.path.join(input_data_path, track)   # Track path
            activity_type, coords_df, wayp_df = extract_information(track_path)  

            if activity_type == 'Senderisme':
                valid_file, error_type = discard_coordinates(zone, coords_df)       # Check if the file is valid
            else:
                valid_file = 0
                error_type = 6      # Error type 6 (discarted because is not a 'Senderisme' activity)

            if valid_file:
                valid_file, error_type, fmm_result, k, r, e = matching_track(model, coords_df, radius=0.001, gps_error=0.001)     # Applying the function

                if valid_file:

                    # Update the waypoints dataframa
                    track_waypoints = waypoints_df_cleaning(wayp_df, track_id, zone)      # Obtain the waypoints dataframe and clean it

                    if track_waypoints is not None:
                        waypoints_df = pd.concat([waypoints_df, track_waypoints])
                        waypoints_df.to_csv(waypoints_df_path, index=False)      # Save the dataframe as csv

                    # Save the 
                    file_output_path = os.path.join(fmm_out_path, str(track_id) + '.csv')
                    save_fmm_output(file_output_path, fmm_result)

                    # Update the output dataframe
                    out_df = pd.concat([out_df, pd.DataFrame({'track_id':[track_id], 'k':[k], 'radius':[r], 'gps_error':[e]})], ignore_index=True)
                    out_df.to_csv(out_df_path, index=False)   # Concatenate the info with the output dataframe and save it
                    log(f"    Found matching path for file {track_id}.", log_path)  # Print information

                else:
                    disc_df = pd.concat([disc_df, pd.DataFrame({'track_id':[track_id], 'zone':[zone], 'error_type':[error_type]})], ignore_index=True)
                    disc_df.to_csv(disc_df_path, index=False)   # Concatenate the info with the discard dataframe and save it
                    log(f"    Error type {error_type} for file {track_id}.", log_path)    # Print information
            else:
                disc_df = pd.concat([disc_df, pd.DataFrame({'track_id':[track_id], 'zone':[zone], 'error_type':[error_type]})], ignore_index=True)
                disc_df.to_csv(disc_df_path, index=False)   # Concatenate the info with the discard dataframe and save it
                log(f"    Error type {error_type} for file {track_id}.", log_path)    # Print information

    log(f'Finished the PROCESSING PART of "{zone}" (Fast Map Matching).', log_path)
    log('----------', log_path)
