import numpy as np
import re
import pandas as pd
from pandas import DataFrame

# all functions used in 3_extract_episodes.py
# make_phenotype_label_matrix() function used in 1_extract_subjects.py

# NON-TIMESERIES PREPROCESSING #

gender_map = {'F': 1, 'M': 2}

# icd9 and icd10 codes that correspond to type 2 diabetes mellitus (t2dm)
diagnosis_labels = ['25000','25010','25020','25030','25040','25050','25060','25070','25080','25090',
                    '25002','25012','25022','25032','25042','25052','25062','25072','25082','25092',
                    'E11','E110','E1100','E1101','E111','E1110','E1111','E112','E1121','E1122','E1129',
                    'E113','E1131','E11311','E11319','E1132','E11321','E113211','E113212','E113213',
                    'E113219','E11329','E113291','E113292','E113293','E113299','E1133','E11331','E113311',
                    'E113312','E113313','E113319','E11339','E113391','E113392','E113393','E113399','E1134',
                    'E11341','E113411','E113412','E113413','E113419','E11349','E113491','E113492','E113493',
                    'E113499','E1135','E11351','E113511','E113512','E113513','E113519','E11352','E113521',
                    'E113522','E113523','E113529','E11353','E113531','E113532','E113533','E113539','E11354',
                    'E113541','E113542','E113543','E113549','E11355','E113551','E113552','E113553','E113559',
                    'E11359','E113591','E113592','E113593','E113599','E1136','E1137','E1137X1','E1137X2',
                    'E1137X3','E1137X9','E1139','E114','E1140','E1141','E1142','E1143','E1144','E1149','E115',
                    'E1151','E1152','E1159','E116','E1161','E11610','E11618','E1162','E11620','E11621','E11622',
                    'E11628','E1163','E11630','E11638','E1164','E11641','E11649','E1165','E1169','E118','E119']

# icd9 and icd10 that correspond to a family history of diabetes mellitus
family_history_diagnoses_labels = ['V180','Z833']

# gender encoding; only two genders exist in MIMIC-IV: 'F' and 'M'
def transform_gender(gender_series):
    global gender_map
    return {'Gender': gender_series.apply(lambda x: gender_map[x])}

# merges stays and diagnoses dataframes (generated from reading the extracted CSV files in 1_extract_subjects.py) 
# of a particular subject (cols: stay_id [index], gender, age, height, weight, BMI and 0/1 for each diagnosis code)
def assemble_episodic_data(stays, diagnoses):
    data = {'Icustay': stays.stay_id, 'Age': stays.anchor_age, 'Length of Stay': stays.los,} # dict with keys as column names and values as Series of column values
    data.update(transform_gender(stays.gender)) # add gender column to data dict
    data['Height'] = np.nan # add height column to data dict
    data['Weight'] = np.nan # add weight column to data dict
    data['BMI'] = np.nan # add BMI column to data dict
    data = DataFrame(data).set_index('Icustay') # convert data dict to dataframe and set stay_id as index
    data = data[['Gender', 'Age', 'Height', 'Weight', 'BMI', 'Length of Stay',]]
    return data.merge(extract_diagnosis_labels(diagnoses), left_index=True, right_index=True)

# returns dataframe with stay_id as index and every icd code associated with t2dm as the columns + family history column
def extract_diagnosis_labels(diagnoses):
    global diagnosis_labels
    global family_history_diagnoses_labels
    diagnoses['value'] = 1  # add 'value' column (original cols: subject_id, hadm_id, seq_num, icd_code, icd_version, long_title, stay_id)
    labels = diagnoses[['stay_id', 'icd_code', 'value']].drop_duplicates()\
                      .pivot(index='stay_id', columns='icd_code', values='value').fillna(0).astype(int)
    missing_cols = [l for l in diagnosis_labels if l not in labels.columns]
    missing_data = pd.DataFrame(0, index = labels.index, columns = missing_cols)
    labels = pd.concat([labels, missing_data], axis=1)
    # for l in diagnosis_labels:
    #     if l not in labels:
    #         labels[l] = 0
    labels = labels[diagnosis_labels]
    labels = labels.rename(dict(zip(diagnosis_labels, ['Diagnosis ' + d for d in diagnosis_labels])), axis=1)
    diagnoses['family_history_indicator'] = diagnoses['icd_code'].isin(family_history_diagnoses_labels)
    labels['Family history'] = diagnoses.groupby('stay_id')['family_history_indicator'].any().astype(int).fillna(0)
    return labels

# returns a phenotype label matrix (# rows of stay_id x 1) with stay_id as index and 'label' column indicating whether this stay was diagnosed with t2dm
# note: some stay_ids (and assc. hadm_ids) may not have corresponding diagnoses, so this must be considered when generating the phenotype label matrix
def make_phenotype_label_matrix(diagnoses, stays):
    global diagnosis_labels
    diagnoses['indicator'] = diagnoses['icd_code'].isin(diagnosis_labels)
    phenotype_label = diagnoses.groupby('stay_id')['indicator'].any().astype(int).reset_index()
    phenotype_label.columns = ['stay_id', 'label']
    phenotype_label = stays[['stay_id']].merge(phenotype_label, how='left', left_on='stay_id', right_on='stay_id').fillna(0).astype(int)
    phenotype_label.set_index('stay_id', inplace=True)
    return phenotype_label

# TIMESERIES PREPROCESSING #

# reads ehr_itemid_to_variable_map.csv into var_map dataframe (cols: variable, itemid, mimic_label)
def read_itemid_to_variable_map(fn, variable_column='LEVEL2'):
    var_map = pd.read_csv(fn).fillna('').astype(str)
    var_map.COUNT = var_map.COUNT.astype(int)
    var_map = var_map[(var_map[variable_column] != '') & (var_map.COUNT > 0)]
    var_map.ITEMID = var_map.ITEMID.astype(int)
    var_map = var_map[[variable_column, 'ITEMID', 'MIMIC LABEL']]
    var_map = var_map.rename({variable_column: 'variable', 'MIMIC LABEL': 'mimic_label'}, axis=1)
    var_map.columns = var_map.columns.str.lower()
    return var_map

# merges the events dataframe of a subject with the var_map dataframe on the itemid column
# i.e. only events whose itemid is in the variable map will be kept (i.e. features)
# final cols of events: subject_id, hadm_id, stay_id, charttime, itemid, value, valuenum, variable, mimic_label
def map_itemids_to_variables(events, var_map):
    return events.merge(var_map, left_on='itemid', right_on='itemid') 

# returns a dataframe formed from reading the ehr_variable_ranges.csv file
def read_variable_ranges(fn, variable_column='LEVEL2'):
    columns = [variable_column, 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH']
    
    to_rename = dict(zip(columns, [c.replace(' ', '_') for c in columns]))
    to_rename[variable_column] = 'variable'

    var_ranges = pd.read_csv(fn, index_col=None)
    var_ranges = var_ranges[columns]
    var_ranges.rename(to_rename, axis=1, inplace=True)
    var_ranges.set_index('variable', inplace=True)
    return var_ranges.loc[var_ranges.notnull().all(axis=1)]

# removes outliers and sets values outside of valid range to the valid range for one of the extracted features in events dataframe
def remove_outliers_for_variable(events, variable, ranges):
    if variable not in ranges.index:    # if variable not in ehr_variable_ranges.csv, do nothing
        return events
    idx = (events.variable == variable)
    v = events.value[idx].copy()
    v.loc[v < ranges.OUTLIER_LOW[variable]] = np.nan
    v.loc[v > ranges.OUTLIER_HIGH[variable]] = np.nan
    v.loc[v < ranges.VALID_LOW[variable]] = ranges.VALID_LOW[variable]
    v.loc[v > ranges.VALID_HIGH[variable]] = ranges.VALID_HIGH[variable]
    events.loc[idx, 'value'] = v
    return events.dropna(subset=['value'])

# cleaning functions
# Systolic BP (mmHg): some may be strings 'sbp/dbp' so extract first number
def clean_sbp(df):
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(1))
    return v.astype(float)

# Diastolic BP (mmHg): some may be strings 'sbp/dbp' so extract second number
def clean_dbp(df):
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(2))
    return v.astype(float)

# Glucose (mg/dL): sometimes may have ERROR as value so replace with NaN
def clean_lab(df):
    v = df.value.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan
    return v.astype(float)

# Temperature (C): convert Farenheit to Celsius; some Celsius are > 79 (assume Farenheit)
def clean_temperature(df):
    v = df.value.astype(float).copy()
    idx = df.valuenum.fillna('').apply(lambda s: 'F' in s.lower()) | df.mimic_label.apply(lambda s: 'F' in s.lower()) | (v >= 79)
    v.loc[idx] = (v[idx] - 32) * 5. / 9
    return v

# Weight (kg): convert pounds to kg
def clean_weight(df):
    v = df.value.astype(float).copy()
    idx = df.valuenum.fillna('').apply(lambda s: 'lb' in s.lower()) | df.mimic_label.apply(lambda s: 'lb' in s.lower())
    v.loc[idx] = v[idx] * 0.453592
    return v

# Height (cm): convert inches to cm
def clean_height(df):
    v = df.value.astype(float).copy()
    idx = df.valuenum.fillna('').apply(lambda s: 'in' in s.lower()) | df.mimic_label.apply(lambda s: 'in' in s.lower())
    v.loc[idx] = np.round(v[idx] * 2.54)
    return v

# variables that do not need cleaning: HbA1c (%), HDL (mg/dL), Heart Rate (bpm), LDL (mg/dL), Respiratory rate (insp/min), Triglycerides (mg/dL), Urine output (mL)

clean_fns = {
    'Diastolic blood pressure': clean_dbp,
    'Systolic blood pressure': clean_sbp,
    'Glucose': clean_lab,
    'Temperature': clean_temperature,
    'Weight': clean_weight,
    'Height': clean_height
}

# clean events dataframe by applying cleaning functions to corresponding variables
def clean_events(events):
    global clean_fns
    for var_name, clean_fn in clean_fns.items():
        idx = (events.variable == var_name)
        try:
            events.loc[idx, 'value'] = clean_fn(events[idx])
        except Exception as e:
            import traceback
            print("Exception in clean_events:", clean_fn.__name__, e)
            print(traceback.format_exc())
            print("number of rows:", np.sum(idx))
            print("values:", events[idx])
            exit()
    return events.loc[events.value.notnull()]