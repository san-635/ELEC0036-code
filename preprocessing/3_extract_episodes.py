import argparse
import os
import sys
from tqdm import tqdm

from ehr_processing import read_stays, read_diagnoses, read_events, get_events_for_stay,\
    add_hours_elapsed_to_events
from ehr_processing import convert_events_to_timeseries, get_first_valid_from_timeseries
from ehr_preprocessing import read_itemid_to_variable_map, map_itemids_to_variables, read_variable_ranges, \
    remove_outliers_for_variable, clean_events, assemble_episodic_data

# goal: from each subject's stays.csv, diagnoses.csv and events.csv files, extract
    # (1) non-timeseries : episodic_data df as episode{#}.csv
    # (2) timeseries : episode df as episode{#}_timeseries.csv

parser = argparse.ArgumentParser(description="3_extract_episodes.py")
parser.add_argument('subjects_root_path', type=str, help="'root' directory containing subject sub-dir.")
parser.add_argument('--variable_map_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'ehr_itemid_to_variable_map.csv'),
                    help='CSV containing the item_id-to-variable map that was created using MIMIC IV datatables.')
parser.add_argument('--reference_range_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'ehr_variable_ranges.csv'),
                    help='CSV containing the reference ranges for variables.')
args = parser.parse_args()

# read ehr_itemid_to_variable_map.csv to get the list of variables that form the features
var_map = read_itemid_to_variable_map(args.variable_map_file)
variables = var_map.variable.unique()
# read ehr_variable_ranges.csv to get the reference ranges for each variable from this list
ranges = read_variable_ranges(args.reference_range_file)

# iterate over all subjects
for subject_dir in tqdm(os.listdir(args.subjects_root_path), desc='Iterating over subjects'):
    dn = os.path.join(args.subjects_root_path, subject_dir)
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        continue

    try:
        # read stays.csv, diagnoses.csv and events.csv of this subject into respective dfs
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
        diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
        events = read_events(os.path.join(args.subjects_root_path, subject_dir))
    except:
        sys.stderr.write('Error reading from disk for subject: {}\n'.format(subject_id))
        continue

    # generate a df for this subject w/ index = stay_id and 
    # cols = Gender, Age, Height, Weight, BMI, Diagnosis 25000, Diagnosis 25001, ... , Diagnosis E118, Diagnosis E119, Family history
    episodic_data = assemble_episodic_data(stays, diagnoses)

    # keep only those rows that have the chosen features and clean events df
    events = map_itemids_to_variables(events, var_map)
    for variable in variables:
        events = remove_outliers_for_variable(events, variable, ranges)
    events = clean_events(events)

    # if this subject has no valid events, skip to next subject
    if events.shape[0] == 0:
        continue

    # convert events df into a timeseries df (index: charttime, cols: selected features, stay_id)
    timeseries = convert_events_to_timeseries(events, variables=variables)

    for i in range(stays.shape[0]): # iterate over all episodes (i.e. admits with one stay) of this subject
        stay_id = stays.stay_id.iloc[i]
        intime = stays.intime.iloc[i]
        outtime = stays.outtime.iloc[i]

        # all events recorded during this stay for this subject, i.e., during this episode
        episode = get_events_for_stay(timeseries, stay_id, intime, outtime)
        if episode.shape[0] == 0:   # if no events for this stay, skip to next episode
            continue
        # replace charttime with hours elapsed since intime in episode dataframe
        episode = add_hours_elapsed_to_events(episode, intime).set_index('HOURS').sort_index(axis=0)

        # extract weight (kg), height (cm), BMI (kg/m^2) from episode dataframe and add to episodic_data dataframe
        if stay_id in episodic_data.index:
            episodic_data.loc[stay_id, 'Weight'] = get_first_valid_from_timeseries(episode, 'Weight')
            episodic_data.loc[stay_id, 'Height'] = get_first_valid_from_timeseries(episode, 'Height')
            episodic_data.loc[stay_id, 'BMI'] = episodic_data.loc[stay_id, 'Weight']/((episodic_data.loc[stay_id, 'Height']/100 + 1e-7)**2)

        # save episodic_data dataframe (non-timeseries features) of this subject as 'episode{#}.csv' in its dir
        # cols: Icustay/stay_id [index], Gender, Age, Height, Weight, BMI, Diagnosis 25000, Diagnosis 25001, ... , Diagnosis E118, Diagnosis E119, Family history
        episodic_data.loc[episodic_data.index == stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
                                                                              'episode{}.csv'.format(i+1)),
                                                                 index_label='Icustay')
        
        # save episode dataframe (timeseries features) of this subject as 'episode{#}_timeseries.csv' in its dir
        # cols: Hours [index], Diastolic blood pressure, ..., Urine output, Weight
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries.csv'.format(i+1)),
                       index_label='Hours')