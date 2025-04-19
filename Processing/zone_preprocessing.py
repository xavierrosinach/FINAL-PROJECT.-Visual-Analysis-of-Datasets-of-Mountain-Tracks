import pandas as pd
import os
import osmnx as ox
import shutil
from shapely.geometry import Polygon, LineString
import zipfile
import geopandas as gpd

data_path = '../../Data'
bounds_dict = {"canigo": (2.2, 2.7, 42.4, 42.6), 
               "matagalls": (2.3, 2.5, 41.8, 41.9), 
               "vallferrera": (1.2, 1.7, 42.5, 42.8)}

# Extracts the zip file if it has not been extracted (only mantain the JSON files)
def extract_zip(zone):

    # Define the paths
    zip_path = os.path.join(data_path, 'Raw-Data', zone + '.zip')
    provisional_path = os.path.join(data_path, 'Raw-Data', zone)    # Provisional path to extract the zip file
    extract_path = os.path.join(data_path, 'Input-Data', zone)      # Path to extract only the json files

    if os.path.exists(extract_path):    # Do not enter to the function if the path already exists
        return extract_path
    
    with zipfile.ZipFile(zip_path, 'r') as zip_file:        # Extract the zip file to the provisional path
        zip_file.extractall(provisional_path)

    os.makedirs(extract_path, exist_ok=True)       # Create the extraction path

    for root, _, files in os.walk(provisional_path):        # Walk through the provisional_path and copy .json files
        for file in files:
            if file.endswith('.json'):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(extract_path, file)
                shutil.copy2(source_file, destination_file)     # Copy the files
                
    shutil.rmtree(provisional_path)     # Remove the provisional directory

    return extract_path

# Creates the OpenStreetMap network using edges and nodes
def create_osm_network(zone):

    network_path = os.path.join(data_path, 'OSM-Data', zone)    # Define the network path

    if os.path.exists(network_path):    # Only enter to the function if the path does not exist
        return network_path
        
    os.makedirs(network_path, exist_ok=True)      # Create the path if it has not been created

    x1,x2,y1,y2 = bounds_dict.get(zone)      # Depending on the zone, select the bounds
    boundary_polygon = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])   # Create the bounds polygon
    G = ox.graph_from_polygon(boundary_polygon, network_type='all')     # Create the graph (all type of paths)

    # Create the paths
    filepath_nodes = os.path.join(network_path, "nodes.shp")    # Nodes shapefile
    filepath_edges = os.path.join(network_path, "edges.shp")    # Edges shapefile

    # Convert undirected graph to GDFs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
    gdf_edges["fid"] = gdf_edges.index      # Create an index for each edge

    # Save the nodes and the edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding="utf-8")
    gdf_edges.to_file(filepath_edges, encoding="utf-8")

    return network_path

# Creates output directories and dataframes
def output_structure(zone):
    
    output_path = os.path.join(data_path, 'Output-Data', zone)      # Define the output path
    dataframes_path = os.path.join(output_path, 'Dataframes')   # Directory with all the dataframes
    fmm_out_path = os.path.join(output_path, 'FMM-Output')      # FMM output
    cleaned_tracks_path = os.path.join(output_path, 'Cleaned-Output')    # Cleaned Output
    logs_path = os.path.join(output_path, 'Logs')

    os.makedirs(dataframes_path, exist_ok=True)     # Create directory
    os.makedirs(fmm_out_path, exist_ok=True)    # Create directory
    os.makedirs(cleaned_tracks_path, exist_ok=True)  # Create directory
    os.makedirs(logs_path, exist_ok=True)

    # Create output and discarded files dataframes
    disc_df_path = os.path.join(dataframes_path, 'discard_files.csv')
    out_df_path = os.path.join(dataframes_path, 'output_files.csv')

    # Discarded files dataframe creation
    if os.path.exists(disc_df_path):
        disc_df = pd.read_csv(disc_df_path)
    else:   # If it does not exist, we create if
        disc_df = pd.DataFrame(columns=['track_id','zone','error_type'])
        disc_df.to_csv(disc_df_path, index=False)   # Save the dataframe as csv

    # Output files dataframe creation
    if os.path.exists(out_df_path):
        out_df = pd.read_csv(out_df_path)
    else:   # If it does not exist, we create it
        out_df = pd.DataFrame(columns=['track_id','k','radius','gps_error'])
        out_df.to_csv(out_df_path, index=False)

    return output_path, dataframes_path, fmm_out_path, cleaned_tracks_path, disc_df, disc_df_path, out_df, out_df_path, logs_path

# Reads the edges shapefile and cleans it for a future processing
def create_edges_df(osm_data_path, dataframes_path):

    edges_df_path = os.path.join(dataframes_path, 'edges.csv')      # Cleaned edges csv path

    if os.path.exists(edges_df_path):   # If the dataframe exists, read it
        edges_df = pd.read_csv(edges_df_path)
        return edges_df_path, edges_df
    
    edges_df = gpd.read_file(os.path.join(osm_data_path, 'edges.shp'))      # Load the edges dataframe with geopandas
    edges_df = edges_df[['u','v','geometry']]       # Select only needed columns

    # Apply order to u and v to avoid duplicated edges
    edges_df[['u', 'v']] = edges_df.apply(lambda row: sorted([row['u'], row['v']]), axis=1, result_type='expand')   # Apply order
    edges_df = edges_df.drop_duplicates(subset=['u','v'])   # Drop duplicated values

    # Sort the columns depending on u, and create an edge id
    edges_df = edges_df.sort_values(by='u').reset_index(drop=True)
    edges_df['id'] = range(1, len(edges_df) + 1)

    # Add empty rows for a future post-processing
    edges_df['total_tracks'] = 0    # Count of the total tracks that pass through that edge
    edges_df['list_tracks'] = [[] for _ in range(len(edges_df))]  # List with the tracks ids that pass through that edge
    edges_df['weight'] = 0      # Normalized count

    # Reorder the dataframe
    edges_df = edges_df[['id','u','v','geometry','total_tracks','list_tracks','weight']]

    # Save the edges dataframe as csv
    edges_df.to_csv(edges_df_path, index=False)

    return edges_df_path, edges_df

# Creates or reads the final output dataframe
def create_cleaned_output_df(dataframes_path):

    clean_out_df_path = os.path.join(dataframes_path, 'cleaned_out.csv')  # Define the path

    # Cleaned output files dataframe creation
    if os.path.exists(clean_out_df_path):
        clean_out_df = pd.read_csv(clean_out_df_path)
    else:   # If it does not exist, we create it
        clean_out_df = pd.DataFrame(columns=['track_id','url','title','user','difficulty','distance','elev_gain','elev_loss','date',
                                             'year','month','weekday','season','min_temp','max_temp','weather_condition',
                                             'total_time', 'avg_pace', 'first_lat', 'first_lon','last_lat','last_lon'])
        clean_out_df.to_csv(clean_out_df_path, index=False)

    return clean_out_df_path, clean_out_df

# Create of read all the waypoints dataframe
def create_waypoints_df(dataframes_path):

    waypoints_df_path = os.path.join(dataframes_path, 'waypoints.csv')  # Define the path
    
    # Dataframe creation
    if os.path.exists(waypoints_df_path):
        waypoints_df = pd.read_csv(waypoints_df_path)
    else:   # If it does not exist, we create it
        waypoints_df = pd.DataFrame(columns=['waypoint_id','track_id','lat','lon','elev','type','url'])
        waypoints_df.to_csv(waypoints_df_path, index=False)

    return waypoints_df_path, waypoints_df

# Full zone preprocessing
def zone_preprocessing(zone):

    input_data_path = extract_zip(zone)     # Extract the zip file
    osm_data_path = create_osm_network(zone)        # Create the OpenStreetMap network
    output_path, dataframes_path, fmm_out_path, cleaned_tracks_path, disc_df, disc_df_path, out_df, out_df_path, logs_path  = output_structure(zone)    # Create the output directories and read dataframes
    edges_df_path, edges_df = create_edges_df(osm_data_path, dataframes_path)   # Clean the edges dataframe for a future processing
    clean_out_df_path, clean_out_df = create_cleaned_output_df(dataframes_path)     # Obtain the cleaned output dataframe
    waypoints_df_path, waypoints_df = create_waypoints_df(dataframes_path)      # Obtain the waypoints dataframe

    return input_data_path, osm_data_path, output_path, dataframes_path, fmm_out_path, cleaned_tracks_path, disc_df, disc_df_path, out_df, out_df_path, edges_df_path, edges_df, clean_out_df_path, clean_out_df, waypoints_df_path, waypoints_df, logs_path