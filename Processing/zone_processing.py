import pandas as pd
import os
import shutil
import zipfile
import signal
import json
import multiprocessing
import numpy as np
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import Polygon, LineString
from shapely.wkt import loads
from geopy.distance import geodesic
from fmm import Network,NetworkGraph,UBODTGenAlgorithm,UBODT,FastMapMatch, FastMapMatchConfig

# Initialize transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

# Define the bounds for each zone
bounds_canigo = (2.2, 2.7, 42.65, 42.35)
bounds_matagalls = (2.3, 2.5, 41.85, 41.8) 
bounds_vallferrera = (1.3, 1.4, 42.65, 42.55)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out")

def run_fmm_match(model, track_wkt, k, r, e, queue):
    try:
        config = FastMapMatchConfig(k, r, e)
        result = model.match_wkt(track_wkt, config)
        queue.put((True, 0, result))  # Indicating success
    except Exception as e:
        queue.put((False, str(e), None))  # Capture any other exceptions

def safe_match(model, track_wkt, k, r, e, timeout=60):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_fmm_match, args=(model, track_wkt, k, r, e, queue))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()  # Kill the process if it hangs
        process.join()
        return False, 7, None  # Segmentation fault detected

    if not queue.empty():
        return queue.get()

    return False, 7, None  # If queue is empty, assume a failure

# Extract the information of the json file as a dictionary - waypoints and coordinates as dataframes
def extract_information(json_path):

    # Load JSON data from the file
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Extract relevant information safely
    return {
        "url": data.get("url"),
        "user": data.get("user"),
        "track": data.get("track"),
        "title": data.get("title"),
        "date_up": data.get("date-up"),
        "date_track": data.get("date-track"),
        "activity_type": data.get("activity", {}).get("type"),  # Extracting "activity.type"
        "activity_name": data.get("activity", {}).get("name"),  # Extracting "activity.name"
        "difficulty": data.get("difficulty"), 
        "waypoints": pd.DataFrame(data.get("waypoints", [])),
        "coordinates": pd.DataFrame(data["coordinates"], columns=["Longitude", "Latitude", "Elevation", "Unknown"])}

# Function to check if we need to discard the file depending on the coordinates df
def discard_coordinates(zone, coordinates_df, min_coordinates=100, max_distance=300, min_total_distance=1000):

    if zone == 'canigo':
        bounds = bounds_canigo
    elif zone == 'matagalls':
        bounds = bounds_matagalls
    elif zone == 'vallferrera':
        bounds = bounds_vallferrera

    total_distance = 0.0    # Initialize a value for the total distance

    # Check if all coordinates are between the bounds
    all_inside_canigo = (
        (coordinates_df["Longitude"] >= bounds[0]) & (coordinates_df["Longitude"] <= bounds[1]) & 
        (coordinates_df["Latitude"] >= bounds[3]) & (coordinates_df["Latitude"] <= bounds[2])
    ).all()

    if not all_inside_canigo:
        return False, 1, total_distance

    # Discard the track if it has less than min_coordinates
    if len(coordinates_df) < min_coordinates:
        return False, 2, total_distance 

    for i in range(len(coordinates_df) - 1):  # Stop at the second last row
        lon1, lat1 = coordinates_df.loc[i, 'Longitude'], coordinates_df.loc[i, 'Latitude']
        lon2, lat2 = coordinates_df.loc[i + 1, 'Longitude'], coordinates_df.loc[i + 1, 'Latitude']
        
        # Compute distance using geopy
        distance = geodesic((lat1, lon1), (lat2, lon2)).meters

        # Check if the distance between two points is less than the maximum
        if distance > max_distance:
            return False, 3, total_distance

        # Sum the distance at the total distance
        total_distance += distance

    # Check if the total distance is greater than the minimum
    if total_distance < min_total_distance:
        return False, 4, total_distance

    return True, 0, total_distance

# Function to obtain the matching track - training process
def matching_track_training(model, coordinates_df, timeout=60):

    # Define the candidates
    k_candidates = [2, 3, 4]
    r_candidates = [2, 3, 4, 5]
    e_candidates = [2, 3, 4, 5]

    # Get the track wkt
    line = LineString(zip(coordinates_df["Longitude"], coordinates_df["Latitude"]))
    track_wkt = line.wkt

    # Initial error (infinite)
    minimum_error = np.inf
    best_result = None
    best_k = 0
    best_r = 0
    best_e = 0

    # Try for all candidates
    for k in k_candidates:
        for r in r_candidates:
            for e in e_candidates:
                valid, error_type, result = safe_match(model, track_wkt, k, r, e, timeout)

                if not valid:
                    if error_type == 7:  # Segmentation fault detected
                        continue
                    else:
                        continue

                mean_error = np.mean([c.error for c in result.candidates])

                if mean_error < min_error:
                    min_error, best_result, best_k, best_r, best_e = mean_error, result, k, r, e

    return (True, 0, best_result, best_k, best_r, best_e) if best_result else (False, 5, None, 0, 0, 0)

# Function to obtain the matching track - training process
def matching_track_test(model, coordinates_df, radius, gps_error, timeout=60):

    k_candidates = [2, 3, 4]

    line = LineString(zip(coordinates_df["Longitude"], coordinates_df["Latitude"]))
    track_wkt = line.wkt

    for k in k_candidates:
        valid, error_type, result = safe_match(model, track_wkt, k, radius, gps_error, timeout)

        if valid:
            return True, 0, result
        elif error_type == 7:  # Segmentation fault detected
            continue

    return False, 6, None

# Function to save the coordinates that have more than 5 meters of error
def out_track_coordinates(track_id, zone, fmm_output_df, no_osm_df, no_osm_df_path, max_error = 0.005):
        
    # Obtain the filtered dataframe - only the original coordinates    
    filtered_df = fmm_output_df[fmm_output_df['error'] >= max_error][['Longitude', 'Latitude']]

    # Add the 'track_id' and 'zone' columns
    filtered_df["track_id"] = track_id
    filtered_df["zone"] = zone

    # Rename columns to match no_osm_df
    filtered_df = filtered_df.rename(columns={"Longitude": "longitude", "Latitude": "latitude"})

    # Append new data to existing no_osm_df and save it as CSV
    no_osm_df = pd.concat([no_osm_df, filtered_df], ignore_index=True)
    no_osm_df.to_csv(no_osm_df_path, index=False)

    return no_osm_df

# Process to output the files
def output_process(file_path, out_df, out_df_path, zone, info, training_process, fmm_result, total_distance, k, r, e, no_osm_df, no_osm_df_path):

    # Concatenate the output dataframe and save it
    out_df = pd.concat([out_df, pd.DataFrame({'track_id': [info['track']], 
                                              'zone': [zone], 
                                              'url': info['url'],
                                              'user': info['user'],
                                              'title': [info['title']],
                                              'date_up': [info['date_up']],
                                              'activity_type': [info['activity_type']],
                                              'activity_type_name': [info['activity_name']], 
                                              'difficulty': [info['difficulty']],
                                              'distance': [np.round(total_distance,2)], 
                                              'k': [k],
                                              'radius': [r],
                                              'gps_error': [e],
                                              'mean_point_error': [np.mean([c.error for c in fmm_result.candidates])],
                                              'is_training': [training_process]})], ignore_index=True)
    out_df.to_csv(out_df_path, index=False)  

    # Concatenate the new coordinates with the old ones
    new_track = fmm_result.pgeom.export_wkt()
    line = loads(new_track)     # Convert to shapely LineString object
    coords = list(line.coords)      # Extract coordinates

    # Create a copy of the coordinates dataframe, and add the coordinates
    coordinates_df = info['coordinates'].copy()
    coordinates_df["CleanLongitude"], coordinates_df["CleanLatitude"] = zip(*coords)

    # Obtain the information of the candidates
    candidates = []
    for c in fmm_result.candidates:
        candidates.append((c.edge_id,c.source,c.target,c.error,c.length,c.offset,c.spdist,c.ep,c.tp))
    candidates_df = pd.DataFrame(candidates, columns=["eid","source","target","error","length","offset","spdist","ep","tp"])

    # Concatenate the dataframes
    coordinates_df = pd.concat([coordinates_df.reset_index(drop=True), candidates_df.reset_index(drop=True)], axis=1)

    # Apply transformation of coordinates
    coordinates_df[['CleanLongitude', 'CleanLatitude']] = coordinates_df.apply(
        lambda row: transformer.transform(row['CleanLongitude'], row['CleanLatitude']), 
        axis=1, result_type="expand")

    # Create the dictionary structure to be saved
    json_data = {"waypoints": info['waypoints'].to_dict(orient='records'),  # Converting DataFrame to a list of dicts
                 "coordinates": coordinates_df.to_dict(orient='records')}  # Converting DataFrame to a list of dicts

    # Write the data to a JSON file
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

    # Proceed with the coordinates that have a big error
    no_osm_df = out_track_coordinates(info['track'], zone, coordinates_df, no_osm_df, no_osm_df_path)

    return out_df, no_osm_df

# Function to process all tracks
def process_all_tracks(zone, raw_path, output_path, model, total_required_training_files=100):

    # Define the paths of the dataframes to store information
    disc_df_path = os.path.join(output_path, 'discard_files.csv')
    out_df_path = os.path.join(output_path, 'output_files.csv')
    no_osm_df_path = os.path.join(output_path, 'no_osm_data.csv')

    # Discarded files dataframe creation
    if os.path.exists(disc_df_path):
        disc_df = pd.read_csv(disc_df_path)
    else:
        disc_df = pd.DataFrame(columns=['track_id', 'zone', 'error_type'])

    # Output files dataframe creation
    if os.path.exists(out_df_path):
        out_df = pd.read_csv(out_df_path)
    else:
        out_df = pd.DataFrame(columns=['track_id', 'zone', 'url', 'user', 'title', 'date_up', 'activity_type', 'activity_type_name', 
                                       'difficulty', 'distance', 'k', 'radius', 'gps_error', 'mean_point_error', 'is_training'])
        
    # No-OSM data dataframe creation
    if os.path.exists(no_osm_df_path):
        no_osm_df = pd.read_csv(no_osm_df_path)
    else:
        no_osm_df = pd.DataFrame(columns=['track_id', 'zone', 'longitude', 'latitude'])

    # Get a list with the discarded files and the files already processed
    disc_files = disc_df['track_id'].unique().tolist()
    out_files = out_df['track_id'].unique().tolist()
    processed_files = list(set(disc_files + out_files))

    # Check if we need to proceed with the training or not
    training_output = out_df[out_df['is_training'] == 1]  
    files_to_train = total_required_training_files - len(training_output)

    # Get the best radius and gps_error and identify if we need to proceed with the training
    if files_to_train == 0:
        training_process = 0 
        mean_radius = out_df['radius'].mean()
        mean_gps_error = out_df['gps_error'].mean()
        print(f"TEST PROCESS. Parameters: radius={mean_radius}, gps_error={mean_gps_error}.")
    else:
        training_process = 1
        mean_radius = 0     # Initialize
        mean_gps_error = 0      # Initialize
        print(f"TRAINING PROCESS. Total files to train: {files_to_train} of {total_required_training_files}.")

    # Iterate through all files
    for file in os.listdir(raw_path):

        # Obtain the track id
        track_id = file.split('.')[0]

        # Check if the file is already processed or not
        if int(track_id) not in processed_files:

            print('---')
            print(f'Processing file {track_id}.')

            # Get the path of the file and obtain all the information
            file_path = os.path.join(raw_path, file)
            info = extract_information(file_path)

            # Check if we need to discard the coordinates
            valid_file, error_type, total_distance = discard_coordinates(zone, info['coordinates'])

            # If the file is valid, we will process with the matching
            if valid_file:

                # Depending if one function or another, go to the matching track function
                if training_process:
                    valid_file, error_type, fmm_result, k, r, e = matching_track_training(model, info['coordinates'])
                else:
                    # For the first time we pass to the training, set the configuration
                    if mean_radius == 0:
                        mean_radius = out_df['radius'].mean()
                        mean_gps_error = out_df['gps_error'].mean()
                        training_process = 0
                    
                    # Test matching track processing
                    valid_file, error_type, fmm_result = matching_track_test(model, info['coordinates'], mean_radius, mean_gps_error)

                # If the file is valid proceed with the output process
                if valid_file:
                    print(f"    Found matching path for file {track_id}. Parameters k={k}, r={r}, e={e}")
                    file_output_path = os.path.join(output_path, file)      # Define the output path
                    out_df, no_osm_df = output_process(file_output_path, out_df, out_df_path, zone, info, training_process, fmm_result, total_distance, k, r, e, no_osm_df, no_osm_df_path)
                
                else:
                    # Concatenate the info with the discard dataframe and save it
                    disc_df = pd.concat([disc_df, pd.DataFrame({'track_id': [info['track']], 
                                                            'zone': [zone], 
                                                            'error_type': error_type})], ignore_index=True)
                    disc_df.to_csv(disc_df_path, index=False)  

            else:
                # Concatenate the info with the discard dataframe and save it
                disc_df = pd.concat([disc_df, pd.DataFrame({'track_id': [info['track']], 
                                                        'zone': [zone], 
                                                        'error_type': error_type})], ignore_index=True)
                disc_df.to_csv(disc_df_path, index=False)  

# Function to extract the JSON files of the zip file
def extract_zip(zone, extract_path):

    # Define the path of the zip file
    raw_zip_path = f'../Data/Raw-Data/{zone}.zip'

    # Provisional path to extract the zip
    provisional_path = f'../Data/Raw-Data/provisional_{zone}'

    # If the zip file does not exist, break
    if not os.path.exists(raw_zip_path):
        print(f"The file {zone}.zip does not exist or is not in the correct directory.")
        return
    
    # Extract the zip file to the provisional path
    with zipfile.ZipFile(raw_zip_path, 'r') as zip_file:
        zip_file.extractall(provisional_path)
    
    # Ensure extract_path exists
    os.makedirs(extract_path, exist_ok=True)

    # Walk through the provisional_path and copy .json files
    for root, _, files in os.walk(provisional_path):
        for file in files:
            if file.endswith('.json'):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(extract_path, file)
                shutil.copy2(source_file, destination_file)     # Copy the files
                
    # Remove the provisional directory
    shutil.rmtree(provisional_path)

# Creates the network of the zone
def create_network(zone, network_path):

    # Depending on the zone, select the bounds
    if zone == 'canigo':
        bounds = bounds_canigo
    elif zone == 'matagalls':
        return
    elif zone == 'vallferrera':
        return
    
    # Obtain the graph with ox and save it
    x1,x2,y1,y2 = bounds
    boundary_polygon = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
    G = ox.graph_from_polygon(boundary_polygon, network_type='all')

    # Create the directory
    os.makedirs(network_path)

    # Create the paths
    filepath_nodes = os.path.join(network_path, "nodes.shp")
    filepath_edges = os.path.join(network_path, "edges.shp")

    # Convert undirected graph to GDFs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
    gdf_edges["fid"] = gdf_edges.index      # Create an index for each edge

    # Save the nodes and the edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding="utf-8")
    gdf_edges.to_file(filepath_edges, encoding="utf-8")

# Main function
def main_processing(zone):

    # Define all the paths for the zone
    raw_path = f'../Data/Raw-Data/{zone}'
    output_path = f'../Data/Output/{zone}'
    network_path = f'../Data/OSM-Data/{zone}'

    # Create the output path if it does not exist
    os.makedirs(output_path, exist_ok=True)

    # Extract the zip file if it does not exist
    if not os.path.exists(raw_path):
        extract_zip(zone, raw_path)

    # Crete the network if it is not created
    if not os.path.exists(network_path):
        create_network(zone, network_path)

    # Obtain the network of the zone
    network = Network(os.path.join(network_path, 'edges.shp'), "fid", "u", "v")
    graph = NetworkGraph(network)

    # Create the UDOBT file
    udobt_path = os.path.join(network_path, 'udobt.txt')
    if not os.path.exists(udobt_path):      # If it is not created
        ubodt_gen = UBODTGenAlgorithm(network,graph)
        status = ubodt_gen.generate_ubodt(udobt_path, 4, binary=False, use_omp=True)
    ubodt = UBODT.read_ubodt_csv(udobt_path)    # If it is created, read it

    # Create the model
    model = FastMapMatch(network,graph,ubodt)

    # Process all the files
    process_all_tracks(zone, raw_path, output_path, model)
    
main_processing(zone='canigo')