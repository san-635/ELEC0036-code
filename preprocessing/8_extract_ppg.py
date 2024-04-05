# adapted from https://wfdb.io/mimic_wfdb_tutorials/tutorials.html
import argparse
from ppg_extraction import *

parser = argparse.ArgumentParser(description="8_extract_ppg.py")
parser.add_argument('output_path', type=str, help="'ppg_orig' directory where extracted PPG signals should be written.")
args = parser.parse_args()

# goal: extract PPG signals (only if at least 2mins) for each subject in MIMIC IV WFDB

database_name = 'mimic4wdb/0.1.0'

# 1/3: Get the paths of all records across all subjects (pxxxxxxxx folders) in the MIMIC IV WFDB database
record_paths = get_records(database_name)

# 2/3: Get the segments in each record that contain PPG signals of sufficient length (i.e., valid records / segments)
req_sig = ['Pleth']         # required PPG signals
req_seg_duration = 2*60     # required segment length in seconds
valid_segments = validate_segments(database_name, record_paths, args.output_path, req_sig, req_seg_duration)
print(f'A total of {len(valid_segments["dir"])} segments, {len(set(valid_segments["record_name"]))} records, and {len(set(valid_segments["subject_id"]))} subjects met all the requirements')

# 3/3: Extract the PPG signals from the valid segments and store them in the output_path/subject_id sub-directory
extract_ppg(valid_segments, args.output_path)