import os
import pandas as pd
from tqdm import tqdm
import shutil

# functions used in 7_preprocess_and_create_cxr.py

def read_metadata(metadata_file, input_path):
    metadata = pd.read_csv(metadata_file)
    metadata = metadata[metadata['ViewPosition'] == 'PA']   # include only posteroanterior frontal CXR images
    cxr_metadata_list = []   # list of dicts to store relevant metadata of each frontal CXR image
    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc='Iterating over CXR metadata'):
        study_time = str(row['StudyTime']).split('.')[0]
        if len(study_time) < 5:
            study_time = '000000'   # reset time if study_time is less than 5 digits (i.e. not possible to extract a valid hour, minute, second)
        cxr_metadata_list.append(
            {
                "subject_id": int(row['subject_id']),
                "study_id": int(row['study_id']),
                "dicom_id": str(row['dicom_id']),
                "study_datetime": pd.to_datetime(str(row['StudyDate']) + ' ' + study_time, format='%Y%m%d %H%M%S').strftime('%Y-%m-%d %H:%M:%S'),
                "image_path": f'{input_path}/p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{int(row["study_id"])}/{row["dicom_id"]}.jpg',
            }
        )
    cxr_metadata = pd.DataFrame(cxr_metadata_list)  # index: subject_id, cols: study_id, dicom_id, study_datetime, image_path
    cxr_metadata.set_index('subject_id', inplace=True)
    return cxr_metadata

def filter_cxr(cxr_metadata, all_stays_path):
    all_stays = pd.read_csv(all_stays_path)
    all_stays = all_stays[['subject_id','hadm_id','stay_id','admittime','dischtime']]
    all_stays.loc[:, 'admittime'] = pd.to_datetime(all_stays['admittime'], format='%Y-%m-%d %H:%M:%S')
    all_stays.loc[:, 'dischtime'] = pd.to_datetime(all_stays['dischtime'], format='%Y-%m-%d %H:%M:%S')
    all_stays = all_stays.groupby(['subject_id']).agg({'admittime': 'min', 'dischtime': 'max'}).reset_index()
    cxr_subjects = cxr_metadata.merge(all_stays, on='subject_id', how='inner')
    cxr_subjects['study_datetime'] = pd.to_datetime(cxr_subjects['study_datetime'], format='%Y-%m-%d %H:%M:%S')
    cxr_subjects = cxr_subjects[(cxr_subjects['study_datetime'] > (cxr_subjects['admittime'] - pd.Timedelta(days=30))) | (cxr_subjects['study_datetime'] < (cxr_subjects['dischtime'] + pd.Timedelta(days=30)))]
    cxr_subjects = cxr_subjects.sort_values('study_datetime').drop_duplicates(subset='subject_id', keep='first')
    return cxr_subjects[['subject_id','image_path']]

def move_to_partition(cxr_subjects, output_path, partition_split_path):
    partition_df = pd.read_csv(partition_split_path)
    subjects = partition_df['<subject_id>_<stay_id>'].apply(lambda x: int(x.split('_')[0]))
    stays = partition_df['<subject_id>_<stay_id>'].apply(lambda x: int(x.split('_')[1]))
    num_set = set()
    num_episodes = 0
    # iterate over all subjects in filtered_metadata.csv
    for _,row in tqdm(cxr_subjects.iterrows(), total=cxr_subjects.shape[0], desc="Moving filtered CXR images to 'ehr_root' partitions"):
        # identify the partition/s in ehr_root that this subject belongs to       
        subject_matches = partition_df[subjects == row['subject_id']].partition
        stay_matches = stays[subjects == row['subject_id']]
        # copy the CXR image of this subject to all its episode sub-dirs. with dicom_id as filename
        if len(subject_matches) > 0:
            for idx, match in subject_matches.items():
                episode_name = f'{row["subject_id"]}_{stay_matches[idx]}'
                if match == 'train':
                    output_episode_dir = os.path.join(output_path, 'train', episode_name)
                elif match == 'dev':
                    output_episode_dir = os.path.join(output_path, 'dev', episode_name)
                elif match == 'test':
                    output_episode_dir = os.path.join(output_path, 'test', episode_name)
                else:
                    continue
                try:
                    shutil.copy(row['image_path'], output_episode_dir)
                    num_episodes += 1
                    num_set.add(row['subject_id'])
                except FileNotFoundError:
                    continue
        else:
            continue    # ignore CXR images of subjects that are not in the EHR dataset (i.e. not in EHR+CXR dataset)
    num_subjects = len(num_set)
    print(f'{num_episodes} episodes of {num_subjects} subjects in ehr_root have a linked CXR image.')