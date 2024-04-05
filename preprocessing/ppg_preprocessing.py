# adapted from https://wfdb.io/mimic_wfdb_tutorials/tutorials.html
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import signal
import shutil

# functions used in 9_preprocess_and_create_ppg.py

# Note: Convert np arrays to lists before saving as CSVs
# Save to CSVs as lists of numbers -> they turn into a single string with []-bracket enclosed comma separated numbers
# Then convert back to np arrays after reading the CSVs (remove [] and replaces commas with '', then convert to np array)

def filter_ppg(input_path, output_path):
    # iterate over all subject sub-directories in the input_path ('ppg_orig' directory)
    directories = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    for subject_id in tqdm(directories, total=len(directories), desc='Filtering PPG signals'):
        subject_dir = os.path.join(input_path, subject_id)
        for file in os.listdir(subject_dir):
            if file.endswith('.csv'):   # iterate over all csv files (valid segments) in the subject sub-directory
                # extract PPG signal (and convert to np array) and its fs from the csv file
                ppg_df = pd.read_csv(os.path.join(subject_dir, file))
                ppg = ppg_df['ppg'][0].strip('[]').replace(',', '')
                ppg = np.fromstring(ppg, sep=' ')
                # print(f'ppg: {ppg}')
                sampling_freq = float(ppg_df['sampling_frequency'][0])
                # create and apply filter
                cutoff_freq = 0.5
                filter = signal.butter(N=3, Wn=cutoff_freq, btype='highpass', analog=False, output='sos', fs=sampling_freq)
                filtered_ppg = signal.sosfiltfilt(filter, ppg)
                # substitute 'ppg' col with filtered PPG signal and save as a csv file in subject's sub-directory in output_path ('ppg_filtered' directory)
                ppg_df['ppg'] = ppg_df['ppg'].astype(object)
                ppg_df.at[0, 'ppg'] = filtered_ppg.tolist()
                file_name, _ = os.path.splitext(file)
                output_subject_dir = os.path.join(output_path, subject_id)
                if not os.path.exists(output_subject_dir):
                    os.makedirs(output_subject_dir)
                ppg_df.to_csv(os.path.join(output_subject_dir, f'{file_name}f.csv'), index=False)

def split_windows(input_path, output_path, window_size_seconds):
    # iterate over all subject sub-directories in the input_path ('ppg_filtered' directory)
    for subject_id in tqdm(os.listdir(input_path), total=len(os.listdir(input_path)), desc='Splitting PPG signals into windows'):
        subject_dir = os.path.join(input_path, subject_id)
        # iterate over all segments in the subject sub-directory
        all_windows_list = []   # list to store DFs with windows of each segment
        for file in os.listdir(subject_dir):
            if file.endswith('f.csv'):
                # extract PPG signal (and convert to np array) and its fs
                ppgf_df = pd.read_csv(os.path.join(subject_dir, file))
                ppgf = ppgf_df['ppg'][0].strip('[]').replace(',', '')
                ppgf = np.fromstring(ppgf, sep=' ')
                sampling_freq = float(ppgf_df['sampling_frequency'][0])
                # split the signal in this segment into 20s windows
                window_size = int(window_size_seconds * round(sampling_freq, 1))  # window size in no. of samples
                windows_list = []    # list to store 20s windows of the PPG signal ppgf (that are converted to lists)
                for i in range(0, len(ppgf), window_size):
                    window = ppgf[i:i+window_size]
                    if len(window) == window_size:  # only append 20s windows to the list
                        windows_list.append(window.tolist()) # convert np array to list
                # store all the windows from this csv file as a DF and then append it to a list
                file_name, _ = os.path.splitext(file)
                col_name = file_name[0:-1]
                # windows_df = pd.DataFrame(windows_list, columns=[col_name]).applymap(lambda x: [x])
                windows_series = pd.Series(windows_list, name=col_name)
                windows_df = windows_series.to_frame()
                all_windows_list.append(windows_df)
        # handle the unequal no. of windows in the segments of subjects when creating the dfs
        # also handle the NaN values in the shorter cols of the dfs in subsquent pre-processing steps
        all_windows_df = pd.concat(all_windows_list, axis=1)    # concatenate the cols of all DFs in seg_windows_list; shorter DFs are padded with NaN values
        # save the windows across all segments as a csv file in subject's sub-directory in output_path ('ppg_windowed' directory)
        output_subject_dir = os.path.join(output_path, subject_id)
        if not os.path.exists(output_subject_dir):
            os.makedirs(output_subject_dir)
        all_windows_df.to_csv(os.path.join(output_subject_dir, 'all_windows.csv'), index=False)

def flat_line_removal(input_path, output_path, threshold=0.01):
    # iterate over all subject sub-directories in the input_path ('ppg_windowed' directory)
    for subject_id in tqdm(os.listdir(input_path), total=len(os.listdir(input_path)), desc='Flat line/peak removal'):
        subject_dir = os.path.join(input_path, subject_id)
        # iterate over the csv file (containing the uncorrupt 20s PPG signal windows of every segment) for this subject
        for file in os.listdir(subject_dir):
            if file.endswith('windows.csv'):
                all_windows_df = pd.read_csv(os.path.join(subject_dir, file))
                all_uncorrupted_list = []    # list to store DFs with uncorrupted windows of each segment
                # iterate over all segments (cols) of this subject
                for seg_name in all_windows_df.columns:
                    uncorrupted_list = []    # list to store uncorrupted windows in this segment
                    # iterate over all 20s PPG signal windows (rows) in this segment
                    for _,row in all_windows_df[[seg_name]].dropna().iterrows():
                        # extract the 20s PPG signal window (and convert to np array)
                        window = row.iloc[0].strip('[]').replace(',', '')
                        window = np.fromstring(window, sep=' ')
                        # flat line or flat peak detection using the difference between consecutive samples
                        diffs = np.abs(np.diff(window))
                        flat_points = np.sum(diffs < threshold)
                        if flat_points / len(window) < 0.20:     # if <20% of the window is flat, then it is useful - include it
                            uncorrupted_list.append(window.tolist()) # convert the 20s PPG signal window from np array to list
                    uncorrupted_series = pd.Series(uncorrupted_list, name=seg_name)
                    uncorrupted_df = uncorrupted_series.to_frame()
                    all_uncorrupted_list.append(uncorrupted_df)
                all_uncorrupted_df = pd.concat(all_uncorrupted_list, axis=1)
                # save the uncorrupted windows across all segments as a csv file in subject's sub-directory in output_path ('ppg_uncorrupt' directory)
                output_subject_dir = os.path.join(output_path, subject_id)
                if not os.path.exists(output_subject_dir):
                    os.makedirs(output_subject_dir)
                all_uncorrupted_df.to_csv(os.path.join(output_subject_dir, 'all_uncorrupted_windows.csv'), index=False)

def create_ppg_dataset(input_path, output_path):
    # iterate over all subject sub-directories in the input_path ('ppg_uncorrupt' directory)
    samples = 0
    for subject_id in tqdm(os.listdir(input_path), total=len(os.listdir(input_path)), desc='Creating PPG datasets'):
        subject_dir = os.path.join(input_path, subject_id)
        # iterate over the csv file (containing the uncorrupt 20s PPG signal windows of every segment) for this subject
        for file in os.listdir(subject_dir):
            if file.endswith('uncorrupted_windows.csv'):
                # the last uncorrupted window of the last segment is selected as the subject's PPG dataset
                all_uncorrupted_df = pd.read_csv(os.path.join(subject_dir, file))
                window = None
                for seg_name in all_uncorrupted_df.columns:
                    # iterate over all 20s PPG signal windows (rows) in this segment
                    for _,row in all_uncorrupted_df[[seg_name]].dropna().iterrows():
                        window = row.iloc[0]
                if window is None:
                    continue    # skip to next subject if all windows are corrupted
                else:
                    window_list = window.strip('[]').split(', ')
                    window_list = [float(sample) for sample in window_list]
                    window_df = pd.DataFrame(window_list, columns=['ppg'])
                    # save the selected window as a csv file in subject's sub-directory in output_path ('ppg_dataset' directory)
                    window_df.to_csv(os.path.join(output_path, f'{subject_id}_ppg.csv'), index=False)
                    samples += 1
    print(f'A total of {samples} PPG samples were added to ppg_dataset')

def move_to_partition(input_path, output_path, partition_split_path):
    partition_df = pd.read_csv(partition_split_path)
    num_set = set()
    num_episodes = 0
    # iterate over all subjects' PPG dataset csv files in 'ppg_dataset'
    for ppg_sample in tqdm(os.listdir(input_path), total=len(os.listdir(input_path)), desc="Moving PPG datasets to 'ehr_root' partitions"):
        subject_id = int(ppg_sample.split('_')[0])
        # identify the partition/s in 'ehr_root' that this subject belongs to
        subjects = partition_df['<subject_id>_<stay_id>'].apply(lambda x: int(x.split('_')[0]))
        stays = partition_df['<subject_id>_<stay_id>'].apply(lambda x: int(x.split('_')[1]))
        subject_matches = partition_df[subjects == subject_id].partition
        stay_matches = stays[subjects == subject_id]
        # copy the PPG dataset csv file of this subject to all its episode sub-dirs. with dicom_id as filename
        if len(subject_matches) > 0:
            for idx, match in subject_matches.items():
                if match == 'train':
                    output_episode_dir = os.path.join(output_path, 'train', f'{subject_id}_{stay_matches[idx]}')
                elif match == 'dev':
                    output_episode_dir = os.path.join(output_path, 'dev', f'{subject_id}_{stay_matches[idx]}')
                elif match == 'test':
                    output_episode_dir = os.path.join(output_path, 'test', f'{subject_id}_{stay_matches[idx]}')
                else:
                    continue
                shutil.copyfile(os.path.join(input_path, ppg_sample), os.path.join(output_episode_dir, 'ppg.csv'))
                num_episodes += 1
                num_set.add(subject_id)
        else:
            continue    # ignore PPG datasets of subjects that are not in the EHR dataset (i.e. not in EHR+PPG dataset)
    num_subjects = len(num_set)
    print(f'{num_episodes} episodes of {num_subjects} subjects in ehr_root have a linked PPG sample.')