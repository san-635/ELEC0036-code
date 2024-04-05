import os
import argparse
import pandas as pd
from tqdm import tqdm

# goal: validate events.csv files of all subjects in 'root' by:
    # (1) excluding events with missing/invalid hadm_id or invalid stay_id
    # (2) recovering missing stay_ids of events using their hadm_ids

def is_subject_folder(x):
    return str.isdigit(x)

def main():
    # stats over all subjects
    n_events = 0                   # total number of events
    empty_hadm = 0                 # if hadm_id is missing in events.csv, exclude such events
    no_hadm_in_stay = 0            # if the hadm_id in events.csv is not in stays.csv (i.e., invalid), exclude such events
    no_icustay = 0                 # if stay_id is missing in events.csv, recover it using hadm_id from stays.csv
    recovered = 0                  # recovered stay_ids of events
    could_not_recover = 0          # stay_ids that could not be recovered; should be zero ideally
    icustay_missing_in_stays = 0   # if the stay_id in events.csv is not in stays.csv, exclude such events

    parser = argparse.ArgumentParser(description="2_validate_events.py")
    parser.add_argument('subjects_root_path', type=str, help="'root' directory containing subject sub-dir.")
    args = parser.parse_args()

    subdirectories = os.listdir(args.subjects_root_path)
    subjects = list(filter(is_subject_folder, subdirectories))

    for subject in tqdm(subjects, desc='Iterating over subjects'):
        # read the stays.csv file generated for this subject during 1_extract_subjects.py
        stays_df = pd.read_csv(os.path.join(args.subjects_root_path, subject, 'stays.csv'))

        # assert that there are no subjects with missing stay_id or hadm_id
        assert(not stays_df['stay_id'].isnull().any())
        assert(not stays_df['hadm_id'].isnull().any())

        # assert there are no duplicates of stay_id or hadm_id (i.e., only stay per admission)
        assert(len(stays_df['stay_id'].unique()) == len(stays_df['stay_id']))
        assert(len(stays_df['hadm_id'].unique()) == len(stays_df['hadm_id']))

        # read the events.csv file generated for this subject during 1_extract_subjects.py
        events_df = pd.read_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'))
        n_events += events_df.shape[0]

        # exclude events with missing hadm_id
        empty_hadm += events_df['hadm_id'].isnull().sum()
        events_df = events_df.dropna(subset=['hadm_id'])

        # merge events.csv with stays.csv to recover missing stay_ids of events
        merged_df = events_df.merge(stays_df, left_on=['hadm_id'], right_on=['hadm_id'],
                                    how='left', suffixes=['', '_r'], indicator=True)
        # overlapping cols: for those from events_df '', for those from stays_df '_r' is appended as suffix
        # indicator: '_merge' col is added to indicate whether the row is present in both (both) or only one of the dataframes (left_only or right_only)
        
        # exclude events whose hadm_id in events.csv does not appear in stays.csv, i.e., invalid hadm_id
        no_hadm_in_stay += (merged_df['_merge'] == 'left_only').sum()
        merged_df = merged_df[merged_df['_merge'] == 'both']

        # recover missing stay_id of events using hadm_id in stays.csv
        cur_no_icustay = merged_df['stay_id'].isnull().sum()
        no_icustay += cur_no_icustay
        merged_df.loc[:, 'stay_id'] = merged_df['stay_id'].fillna(merged_df['stay_id_r'])
        recovered += cur_no_icustay - merged_df['stay_id'].isnull().sum()
        
        # exclude events whose stay_id could not be recovered
        could_not_recover += merged_df['stay_id'].isnull().sum()
        merged_df = merged_df.dropna(subset=['stay_id'])

        # exclude events whose stay_id in events.csv does not appear in stays.csv, i.e., invalid stay_id
        icustay_missing_in_stays += (merged_df['stay_id'] != merged_df['stay_id_r']).sum()
        merged_df = merged_df[(merged_df['stay_id'] == merged_df['stay_id_r'])]

        # replace prev events.csv of this subject with the one generated from the validated merged_df dataframe
        to_write = merged_df[['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum']]
        to_write.to_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'), index=False)

    assert(could_not_recover == 0)
    # print stats over all subjects
    print('n_events: {}'.format(n_events))
    print('empty_hadm: {}'.format(empty_hadm))
    print('no_hadm_in_stay: {}'.format(no_hadm_in_stay))
    print('no_icustay: {}'.format(no_icustay))
    print('recovered: {}'.format(recovered))
    print('could_not_recover: {}'.format(could_not_recover))
    print('icustay_missing_in_stays: {}'.format(icustay_missing_in_stays))


if __name__ == "__main__":
    main()