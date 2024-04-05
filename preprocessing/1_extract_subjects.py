import argparse
import numpy as np
import pandas as pd
from ehr_extraction import *
from ehr_preprocessing import make_phenotype_label_matrix

# goal: extract stays.csv, diagnoses.csv and event.csv files for each subject using MIMIC IV v2.2 datatables 
# (patients, admissions, icustays, d_icd_diagnoses, diagnoses_icd, chartevents, labevents, outputevents)

parser = argparse.ArgumentParser(description="1_extract_subjects.py")
parser.add_argument('mimic4_path', type=str, help="Current directory containing MIMIC-IV folders, hosp and icu.")
parser.add_argument('output_path', type=str, help="'root' directory where per-subject data should be written.")
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Datatables from which to read events.',
                    default=['OUTPUTEVENTS', 'CHARTEVENTS', 'LABEVENTS'])
parser.add_argument('--itemids_file', '-i', type=str,default=os.path.join(os.path.dirname(__file__),'ehr_itemid_to_variable_map.csv'),
                    help='CSV containing list of item_ids to keep.')
parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Verbosity in output.')
parser.add_argument('--quiet', '-q', dest='verbose', action='store_false', help='Suspend printing of details.')
parser.set_defaults(verbose=True)
args, _ = parser.parse_known_args()

try:
    os.makedirs(args.output_path)
except:
    pass

# read patients.csv into patients dataframe
patients = read_patients_table(f'{args.mimic4_path}/hosp/patients.csv')

# read admissions.csv into admits dataframe
admits = read_admissions_table(f'{args.mimic4_path}/hosp/admissions.csv')

# read icustays.csv into stays dataframe
stays = read_icustays_table(f'{args.mimic4_path}/icu/icustays.csv')

# read d_icd_diagnoses.csv and diagnoses_icd.csv into diagnoses dataframe
diagnoses = read_icd_diagnoses_table(f'{args.mimic4_path}/hosp')

if args.verbose:
    print('ORIGINAL:\n\tstay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
          stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))

# STAYS PRE-PROCESSING AND SAVE TO 'root/all_stays.csv' #
# step 1/4
stays = remove_icustays_with_transfers(stays)
if args.verbose:
    print('REMOVE ICU TRANSFERS:\n\tstay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
          stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))
# step 2/4
stays = merge_on_subject_admission(stays, admits)
# step 3/4
stays = merge_on_subject(stays, patients)
# step 4/4
stays = filter_admissions_on_nb_icustays(stays)
if args.verbose:
    print('REMOVE SUBSEQUENT STAYS FOR ADMIT WITH MULTIPLE STAYS:\n\tstay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
          stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))
# save pre-processed stays df to CSV file in 'root' (w/ subject_id, hadm_id, stay_id, last_careunit, intime, 
# outtime, los, admittime, dischtime, deathtime, gender, anchor_age, dod)
stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)


# DIAGNOSES PRE-PROCESSING AND SAVE AT 'root/all_diagnoses.csv' #
# step 1/1
diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
# save pre-processed diagnoses df to CSV file (w/ subject_id, hadm_id, seq_num, icd_code, icd_version, long_title, stay_id)
diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)

# GENERATE PHENOTYPE LABELS AND SAVE AT 'root/phenotype_labels.csv' #
make_phenotype_label_matrix(diagnoses, stays).to_csv(os.path.join(args.output_path,'phenotype_labels.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
# (index=True to also save the stay_id col as index)
# make_phenotype_label_matrix(diagnoses, stays).to_csv(os.path.join(args.output_path,'phenotype_labels.csv'), index=True, quoting=csv.QUOTE_NONNUMERIC)

# BREAK UP STAYS AND DIAGNOSES BY SUBJECT AND SAVE TO RESPECTIVE SUB-DIR AS 'stays.csv' AND 'diagnoses.csv' #
subjects = stays.subject_id.unique()    # all subject_ids in stays df (i.e., merged stays+admits+patients dfs)
break_up_stays_by_subject(stays, args.output_path, subjects=subjects)
break_up_diagnoses_by_subject(diagnoses, args.output_path, subjects=subjects)

# BREAK UP EVENT TABLES BY SUBJECT AND SAVE TO RESPECTIVE SUB-DIR AS 'events.csv' #
# features to be extracted from chartevents.csv, labevents.csv, and outputevents.csv (given in ehr_itemid_to_variable_map.csv)
items_to_keep = set([int(itemid) for itemid in pd.read_csv(args.itemids_file)['ITEMID'].unique()]) if args.itemids_file else None
# handle chartevents.csv, labevents.csv and outputevents.csv event tables
for table in args.event_tables:
    read_events_table_and_break_up_by_subject(f'{args.mimic4_path}', table, args.output_path, items_to_keep=items_to_keep, subjects_to_keep=subjects)