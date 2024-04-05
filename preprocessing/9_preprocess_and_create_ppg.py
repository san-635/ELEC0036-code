import argparse
from ppg_preprocessing import *

parser = argparse.ArgumentParser(description="9_preprocess_and_create_ppg.py")
parser.add_argument('input_path', type=str, help="'ppg_orig' directory where extracted PPG signals are stored.")
parser.add_argument('filter_output_path', type=str, help="'ppg_filtered' directory where filtered PPG signals should be stored.")
parser.add_argument('windows_output_path', type=str, help="'ppg_windowed' directory where 20s PPG windows should be stored.")
parser.add_argument('uncorrupt_output_path', type=str, help="'ppg_uncorrupt' directory where uncorrupted PPG windows should be stored.")
parser.add_argument('dataset_output_path', type=str, help="'ppg_dataset' directory where PPG datasets should be stored.")
parser.add_argument('root_output_path', type=str, help="'ehr_root' directory where PPG datasets should be stored in respective partitions.")
parser.add_argument('--ehr_partition_split', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'ehr_partition.csv'),
                    help="CSV containing the testset split of the EHR dataset.")
args, _ = parser.parse_known_args()

# goal: preprocess the extracted PPG signals and create PPG datasets for each subject in MIMIC IV WFDB

database_name = 'mimic4wdb/0.1.0'

# 1/5: Apply 3rd order 0.5Hz high-pass Butterworth filter to remove baseline wandering due to respiration
filter_ppg(args.input_path, args.filter_output_path)

# 2/5: Split the filtered PPG signals into 20s windows with no overlap
split_windows(args.filter_output_path, args.windows_output_path, window_size_seconds=20)

# 3/5: Detect and exclude windows with >20% flat lines / flat peaks
flat_line_removal(args.windows_output_path, args.uncorrupt_output_path, threshold=0.01)

# 4/5: Create the PPG datasets for each subject
create_ppg_dataset(args.uncorrupt_output_path, args.dataset_output_path)

# 5/5: Copy the PPG datasets into ehr_root/train, ehr_root/dev or ehr_root/test, based on ehr_partition.csv
move_to_partition(args.dataset_output_path, args.root_output_path, args.ehr_partition_split)