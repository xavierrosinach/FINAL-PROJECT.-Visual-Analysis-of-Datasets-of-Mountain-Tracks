import os
import json
import folium

def plot_routes_on_map(directory, output_html="map.html"):
    """
    Reads all JSON files from the given directory, extracts coordinates, 
    and plots each route as an independent line on an interactive Folium map.
    """
    route_map = folium.Map(location=[42.625081, 1.388083], zoom_start=13)

    # Color list for different routes
    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray", "black"]
    
    # Counter to cycle through colors
    color_index = 0

    # Loop through each JSON file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):  # Ensure we only process JSON files
            file_path = os.path.join(directory, filename)

            # Load JSON data
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract coordinates
            coordinates = data.get("coordinates", [])
            if coordinates:
                lat_lng_pairs = [(coord[1], coord[0]) for coord in coordinates]  # Folium uses (latitude, longitude)

                # Assign a unique color per route
                route_color = colors[color_index % len(colors)]
                color_index += 1

                # Add the route as a separate line
                folium.PolyLine(lat_lng_pairs, color=route_color, weight=2.5, opacity=0.8, tooltip=filename).add_to(route_map)

                # Set the initial map center dynamically to the first point of the first route
                if len(lat_lng_pairs) > 0 and color_index == 1:
                    route_map.location = lat_lng_pairs[0]

    # Save map to an HTML file
    route_map.save(output_html)
    print(f"Map has been saved to {output_html}")

# Set the directory containing JSON files
directory_path = "../Data/Raw-Data/canigo"  # Change this to your actual path

# Plot all routes on an interactive map
plot_routes_on_map(directory_path)
