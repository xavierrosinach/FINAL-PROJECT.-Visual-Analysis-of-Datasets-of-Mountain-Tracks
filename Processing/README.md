# Processing Pipeline

The processing pipeline reads all JSON documents in the specified area, transforms the tracking using FastMapMatching, and projects the coordinates to the EPSG:2154 system. It also discards those routes where there was an error in the processing. All data (input and output - both discarded and transformed routes) is saved.

Three zones:
* Canigó (*canigo*)
* Matagalls (*matagalls*)
* Vall Ferrera (*vallferrera*)

The library used for performing the map matching between the input track and the path registered in Open Street Map is [FastMapMatching](https://fmm-wiki.github.io/). This requires the road network as input and three configuration parameters:
* Number of candidates (`k`).
* Search radius (`radius`).
* GPS error (`gps_error`).

Below are the steps followed to process the tracks.

## 1. Initialization

Given the input zone, we create (or read) all the necessary folders and files for processing.

1. Creation (or reading) of general folders in the `/Data` directory: `/Raw-Data/zone`, `/Output/zone`, and `/OSM-Data/zone`.
2. Extraction of the ZIP file if it is not already extracted (function `extract_zip()`):
    * Extract all content from the ZIP file to a temporary folder.
    * From the temporary folder, move all JSON files (routes) to the correct folder: `/Data/Output/zone`.
    * Delete the temporary folder created.
3. Creation of the Open Street Map road network for the zone using the `create_network()` function.
    * It is only created if the path does not already exist.
    * It uses the bounds defined at the beginning of the code.
    * Saves all the information to `/Data/OSM-Data/zone`.
4. Creation of the network and graph necessary to create the model for Fast Map Matching.
    * Create the network and graph from the data in `/Data/OSM-Data/zone`. 
    * Generate (or read) the `udobt.txt` file.
    * Create the model based on the network, graph, and `udobt.txt` file.

## 2. Tracks Cleaning

Given the zone with its folders, and the model generated for it, we initialize the processing of all the tracks in that zone. Before starting, however, we create (or read) three CSV documents in which we will store information as we progress with the processing:
* Processed files (`/Data/Output/zone/output_files.csv`): contains a record of those tracks that we have processed, with their information and the parameters used for Fast Map Matching.
* Discarded files (`/Data/Output/zone/discard_files.csv`): contains a record of those tracks that we have discarded, and the reason (identified with an integer) for which they were discarded.
* Coordinates off-road (`/Data/Output/zone/no_osm_data.csv`): contains the coordinates for which Fast Map Matching resulted in an error greater than 5 meters. These are assumed to be coordinates where there is no Open Street Map path registered, but people still pass through.

Once we have the three record dataframes defined, we create a list with all the tracks already processed (both processed and discarded tracks). We will use this to check that the track we want to clean is not already in there.

The next step is to process the document (if it has not been processed in another execution).
1. We perform an initial filter of the route (function `discard_coordinates()`) discarding those tracks that:
    * Have less than the 50% of the coordinates inside the defined bounds. (*Error type = 1*).
    * Have fewer than 100 coordinates. (*Error type = 2*).
    * Have a distance greater than 300 meters between two coordinates. (*Error type = 3*).
    * Have a total distance less than 1000 meters. (*Error type = 4*).
2. We proceed with the map matching algorithm.
    * Fixed values of `radius=100 meters` and `gps_error=100 meters`. 
    * For `k=[2, 3, 4]`.
    * If we find a matching track within 60 seconds, we return that matching track along with the parameters used.
    * If no match is found for all values of `k`, we discard the file. (*Error type = 5*).
3. If we find a path, and we haven't discarded the track at any point, we proceed with the output of the found track.
4. For some tracks, the library cannot process them and ends execution with a *Segmentation Fault*. In these cases, the error is introduced manually. (*Error type = 6*).

## 3. Matching Track Export

This step is carried out if a path has been found with FastMapMatching, either in the training process or in the test process.

1. We concatenate the track information along with the distance, the parameters used, the average error between the points, and a boolean indicating whether we are in a test or training process.
2. We export the result to a dataframe and concatenate it with the previous coordinates under the names `CleanLongitude` and `CleanLatitude`.
3. We also concatenate this dataframe with information about the axes, errors, etc. (Output from FastMapMatching).
4. We transform the coordinates of the new path to the EPSG:2154 coordinate system.
5. We save this dataframe along with the previous waypoint file in the defined output JSON file.
6. We proceed to check those coordinates that are off the Open Street Map path.
    * For each coordinate in the previous path and the new path, we check the error.
    * If the error is greater than 5 meters, we add the coordinates to the previously created dataframe.
    * We save the dataframe in CSV format.
