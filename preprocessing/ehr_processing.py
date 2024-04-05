import numpy as np
import os
import pandas as pd

# functions used in 3_extract_episodes.py

# returns a dataframe from reading stays.csv from 1_extract_subjects.py
def read_stays(subject_path):
    stays = pd.read_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    stays.dod = pd.to_datetime(stays.dod)
    stays.deathtime = pd.to_datetime(stays.deathtime)
    stays.sort_values(by=['intime', 'outtime'], inplace=True)
    return stays

# returns a dataframe from reading diagnoses.csv from 1_extract_subjects.py
def read_diagnoses(subject_path):
    return pd.read_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)

# returns a dataframe from reading events.csv from 1_extract_subjects.py
def read_events(subject_path):
    events = pd.read_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    events = events[events.value.notnull()] # exclude rows where value of variable is missing
    events.charttime = pd.to_datetime(events.charttime)
    events.hadm_id = events.hadm_id.fillna(value=-1).astype(int)    # already removed events with missing hadm_id in 2_validate_events.py
    events.stay_id = events.stay_id.fillna(value=-1).astype(int)    # already recovered missing stay_ids of events in 2_validate_events.py
    events.valuenum = events.valuenum.fillna('').astype(str)    # fill missing valuenum (unit of variable) with empty string
    # events.sort_values(by=['charttime', 'itemid', 'stay_id'], inplace=True)
    return events

# convert an events dataframe into a timeseries dataframe (index: charttime, cols: selected features, stay_id)
def convert_events_to_timeseries(events, variable_column='variable', variables=[]):
    metadata = events[['charttime', 'stay_id']].sort_values(by=['charttime', 'stay_id'])\
                    .drop_duplicates(keep='first').set_index('charttime')
    timeseries = events[['charttime', variable_column, 'value']]\
                    .sort_values(by=['charttime', variable_column, 'value'], axis=0)\
                    .drop_duplicates(subset=['charttime', variable_column], keep='last')
    timeseries = timeseries.pivot(index='charttime', columns=variable_column, values='value')\
                    .merge(metadata, left_index=True, right_index=True)\
                    .sort_index(axis=0).reset_index()
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    return timeseries

# returns events recorded with a specific stay_id (within specific intime and outtime if specified) for a subject
# i.e., an episode of a subject # also removes stay_id column
def get_events_for_stay(timeseries, icustayid, intime=None, outtime=None):
    idx = (timeseries.stay_id == icustayid)
    if intime is not None and outtime is not None:
        idx = idx | ((timeseries.charttime >= intime) & (timeseries.charttime <= outtime))
    timeseries = timeseries[idx]
    del timeseries['stay_id']
    return timeseries

# returns episode dataframe for a subject where charttime is changed to hours elapsed since intime
def add_hours_elapsed_to_events(episode, intime, remove_charttime=True):
    episode = episode.copy()
    episode['HOURS'] = (episode.charttime - intime).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60
    if remove_charttime:
        del episode['charttime']
    return episode

# return the first non-null value of a specified variable from the episode dataframe of a subject
def get_first_valid_from_timeseries(episode, variable):
    if variable in episode:
        idx = episode[variable].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return episode[variable].iloc[loc]
    return np.nan