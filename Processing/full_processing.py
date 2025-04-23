import time
import os
from datetime import datetime
from zone_fmm import zone_fmm
from zone_postprocessing import zone_postprocessing

data_path = '../../Data'

# Function to print the message and append it in the log file
def log(message, log_file=None):
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def zone_full_processing(zone):

    identifier = datetime.now().strftime("%d%m%Y_%H%M")     # Identifier for the log file
    log_path = os.path.join(data_path, 'Output-Data', zone, 'Logs', str(identifier) + '.txt')

    start_time = time.time()
      
    zone_fmm(zone, log_path)
    zone_postprocessing(zone, log_path)

    end_time = time.time()
    elapsed = end_time - start_time
    mins, secs = divmod(elapsed, 60)

    print(f"Processing for zone '{zone}' completed in {int(mins)} min {secs:.1f} sec.")

# zone_full_processing('vallferrera')
# zone_full_processing('canigo')
# zone_full_processing('matagalls')
