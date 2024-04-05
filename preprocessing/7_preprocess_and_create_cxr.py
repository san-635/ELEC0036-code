import argparse
import os

from cxr_preprocessing import *

# goal: pre-process the resized MIMIC CXR images and create a dataset of CXR images for each subject

parser = argparse.ArgumentParser(description="7_preprocess_and_create_cxr.py")
parser.add_argument('input_path', type=str, help="'cxr_dataset' directory where all resized CXR images are stored.")
parser.add_argument('output_path', type=str, help="'ehr_root' directory where the filtered CXR images should be stored in respective partitions.")
parser.add_argument('--metadata_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'cxr_mimic_metadata.csv'),
                    help='CSV containing metadata associated with MIMIC CXR JPG.')
parser.add_argument('--all_stays', type=str,
                    default=os.path.join(os.path.dirname(__file__),'root','all_stays.csv'),
                    help='CSV containing all stays.')
parser.add_argument('--ehr_partition_split', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'ehr_partition.csv'),
                    help="CSV containing the testset split of the EHR dataset.")
args = parser.parse_args()

# 1/3: Include PA frontal CXR images only and then extract relevant metadata of each CXR image
cxr_metadata = read_metadata(args.metadata_file, args.input_path)
cxr_metadata.to_csv(os.path.join(args.input_path, 'all_metadata_.csv'), index=True)

# 2/3: Extract valid CXR image for each subject (exclude CXR images taken a month before earliest admittime or a month after latest dischtime
# and exclude subsequent CXR images if multiple exist for a subject)
cxr_metadata_subjects = filter_cxr(cxr_metadata, args.all_stays)
cxr_metadata_subjects.to_csv(os.path.join(args.input_path, 'filtered_metadata_.csv'), index=False)

# 3/3: Save the extracted CXR images to the respective subject sub-dir. in cxr_dataset
move_to_partition(cxr_metadata_subjects, args.output_path, args.ehr_partition_split)