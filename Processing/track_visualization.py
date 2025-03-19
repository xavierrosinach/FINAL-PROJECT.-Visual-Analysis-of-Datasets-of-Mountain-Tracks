import json
import matplotlib.pyplot as plt

zone = 'canigo'
track_id = '100049704'

# Load JSON data
file_path = f"../Data/Output/{zone}/{track_id}.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract coordinates
coordinates = data["coordinates"]

# Separate the two sets of longitude and latitude values
# longitudes = [point["Longitude"] for point in coordinates]
# latitudes = [point["Latitude"] for point in coordinates]
clean_longitudes = [point["CleanLongitude"] for point in coordinates]
clean_latitudes = [point["CleanLatitude"] for point in coordinates]

# Plotting
plt.figure(figsize=(10, 6))
# plt.plot(longitudes, latitudes, label="Original Path", color="blue", linestyle="--")
plt.plot(clean_longitudes, clean_latitudes, label="Cleaned Path", color="red", linestyle="-")

# Labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Comparison of Original and Cleaned GPS Paths")
plt.legend()
plt.grid()

# Show the plot
plt.show()
