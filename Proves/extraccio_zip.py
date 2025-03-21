import zipfile
import shutil
import os

# Define the path of the zip file
raw_zip_path = f'../../Data/canigo.zip'

# Provisional path to extract the zip
provisional_path = f'../../Data/provisional_canigo'

# Extract the zip file to the provisional path
with zipfile.ZipFile(raw_zip_path, 'r') as zip_file:
    zip_file.extractall(provisional_path)

# Ensure extract_path exists
extract_path = f'../../Data/canigo'
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