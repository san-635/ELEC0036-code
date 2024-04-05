import csv
import os
import pandas as pd
from tqdm import tqdm

# functions used in 1_extract_subjects.py

# READ CSVs INTO DATAFRAMES #

# read patients.csv into pats DataFrame (w/ subject_id, gender, anchor_age, dod)
def read_patients_table(path):
    pats = pd.read_csv(path)
    columns = ['subject_id', 'gender', 'anchor_age', 'dod']  
    pats = pats[columns]
    pats.dod = pd.to_datetime(pats.dod)
    return pats

# read admissions.csv into admits DataFrame (w/ subject_id, hadm_id, admittime, dischtime, deathtime)
def read_admissions_table(path):
    admits = pd.read_csv(path)
    admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime']]
    admits.admittime = pd.to_datetime(admits.admittime)
    admits.dischtime = pd.to_datetime(admits.dischtime)
    admits.deathtime = pd.to_datetime(admits.deathtime)
    return admits

# read icustays.csv into stays DataFrame (w/ subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime, los)
def read_icustays_table(path):
    stays = pd.read_csv(path)
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    return stays

# read d_icd_diagnoses.csv and diagnoses_icd.csv into diagnoses DataFrame (w/ subject_id, hadm_id, seq_num, icd_code, icd_version, long_title)
def read_icd_diagnoses_table(path):
    codes = pd.read_csv(f'{path}/d_icd_diagnoses.csv')
    codes = codes[['icd_code', 'long_title']]

    diagnoses = pd.read_csv(f'{path}/diagnoses_icd.csv')
    diagnoses = diagnoses.merge(codes, how='inner', left_on='icd_code', right_on='icd_code')
    diagnoses[['subject_id', 'hadm_id', 'seq_num']] = diagnoses[['subject_id', 'hadm_id', 'seq_num']].astype(int)
    return diagnoses

# read a particular events table and return the next row
def read_events_table_by_row(mimic4_path, table):
    nb_rows = {'chartevents': 313645063, 'labevents': 118171367, 'outputevents': 4234967}  # mimic4 v2.2
    # nb_rows = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219} # mimic3
    csv_files = {'chartevents': 'icu/chartevents.csv', 'labevents': 'hosp/labevents.csv', 'outputevents': 'icu/outputevents.csv'}
    reader = csv.DictReader(open(os.path.join(mimic4_path, csv_files[table.lower()]), 'r'))
    for i, row in enumerate(reader):
        if 'stay_id' not in row:
            row['stay_id'] = ''
        yield row, i, nb_rows[table.lower()]

# STAYS DATAFRAME PRE-PROCESSING #

# step 1/4: exclude stays with transfers
def remove_icustays_with_transfers(stays):
    stays = stays[(stays.first_careunit == stays.last_careunit)]
    return stays[['subject_id', 'hadm_id', 'stay_id', 'last_careunit', 'intime', 'outtime', 'los']]

# step 2/4: merge stays with admissions DataFrame (exclude admissions with no stays because every entry of stays has a corresponding entry in admissions, but not vice versa)
def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

# step 3/4: merge (stays+admissions) with patients DataFrame
def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['subject_id'], right_on=['subject_id'])

# step 4/4: exclude subsequent stays for admissions with multiple stays
def filter_admissions_on_nb_icustays(stays):
    # to_keep: cols are [1] hadm_id and [2] count of stay_id for each hadm_id, index is reset
    to_keep = stays.groupby('hadm_id').count()[['stay_id']].reset_index()
    # to_keep1: hadm_id col of to_keep df with one stay only
    to_keep1 = to_keep[(to_keep.stay_id == 1)][['hadm_id']]
    # stays_one: stay_id col of stays df with one stay only
    stays_one = stays[stays['hadm_id'].isin(to_keep1['hadm_id'])][['stay_id']]
    # to_keep2: hadm_id col of to_keep df with more than one stay
    to_keep2 = to_keep[(to_keep.stay_id > 1)][['hadm_id']]
    # stays_more_than_max: all cols of stays df with more than one stay
    stays_more_than_max = stays[stays['hadm_id'].isin(to_keep2['hadm_id'])]
    # identify rows containing the first stay for admissions with more than one stay (using the intime of stays)
    stays_more_than_max = stays_more_than_max.loc[stays_more_than_max.groupby('hadm_id')['intime'].idxmin()]
    # stays_more_than_max: stay_id col of such stays (don't use hadm_id as it is not unique for each of the multiple stays)
    stays_more_than_max = stays_more_than_max[['stay_id']]
    # merge stays_one and stays_more_than_max and then merge with stays df to get the final stays df
    to_keep_stays = stays_one.merge(stays_more_than_max, how='outer', left_on='stay_id', right_on='stay_id')
    stays = stays.merge(to_keep_stays, how='inner', left_on='stay_id', right_on='stay_id')
    return stays

# DIAGNOSES DATAFRAME PRE-PROCESSING #

# step 1/1: exclude diagnoses of excluded stays and merge with stays (on subject_id and hadm_id since diagnoses are associated with hadm_id)
# note: a stay can be diagnosed with multiple icd codes, thus #rows(diagnoses) > #rows(stays)
def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates(), how='inner',
                           left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

# BREAK UP DATAFRAMES BY SUBJECT #

# 1/3: stays table
# break up stays df by subject_id
# save each subject's stays df as 'stays.csv' in subject's sub-directory inside 'root' dir 
# if multiple admissions exist for a subject, then sort the rows of their stays df by intime before saving
def break_up_stays_by_subject(stays, output_path, subjects=None):
    subjects = stays.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up stays by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        stays[stays.subject_id == subject_id].sort_values(by='intime').to_csv(os.path.join(dn, 'stays.csv'), index=False)

# 2/3: diagnoses table
# break up diagnoses df by subject_id
# save each subject's diagnoses df as 'diagnoses.csv' in subject's sub-directory inside 'root' dir 
# if multiple admissions exist for a subject, then sort the rows of their stays df by stay_id and then seq_num before saving
def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None):
    subjects = diagnoses.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up diagnoses by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        diagnoses[diagnoses.subject_id == subject_id].sort_values(by=['stay_id', 'seq_num']).to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)

# 3/3: events tables (chartevents/labevents/outputevents)
# return a single CSV file (w/ subject_id, hadm_id, stay_id, charttime, itemid, value, valuenum)
# that includes all event tables for a subject_id
def read_events_table_and_break_up_by_subject(mimic4_path, table, output_path, items_to_keep=None, subjects_to_keep=None):
    obs_header = ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum']
    if items_to_keep is not None:
        items_to_keep = set([str(s) for s in items_to_keep])
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    # function to append rows from a particular events table to the single 'events.csv' file
    def write_current_observations():
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):    # if path does not exist or is not a file
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')    # create an empty events table with obs_header
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL) # append .curr_obs to events table
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []    # reset .curr_obs

    # nb_rows_dict = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219} # mimic3
    nb_rows_dict = {'chartevents': 313645063, 'labevents': 118171367, 'outputevents': 4234967}  # mimic4 v2.2
    nb_rows = nb_rows_dict[table.lower()]

    # iterate through each row of the particular events table and append only those rows that contain one of the features to the single 'events.csv' file
    for row,row_no,_ in tqdm(read_events_table_by_row(mimic4_path,table),total=nb_rows,desc='Processing {} table'.format(table)):
        if (subjects_to_keep is not None) and (row['subject_id'] not in subjects_to_keep):
            continue
        if (items_to_keep is not None) and (row['itemid'] not in items_to_keep):
            continue
        row_out = {'subject_id': row['subject_id'],
                   'hadm_id': row['hadm_id'],
                   'stay_id': '' if 'stay_id' not in row else row['stay_id'],
                   'charttime': row['charttime'],
                   'itemid': row['itemid'],
                   'value': row['valuenum'] if table=='LABEVENTS' else row['value'],
                   'valuenum': row['valueuom']}   # valuenum stores unit of measurement
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['subject_id']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['subject_id']

    if data_stats.curr_subject_id != '':
        write_current_observations()