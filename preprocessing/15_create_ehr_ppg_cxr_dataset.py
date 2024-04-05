import os
import argparse
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm
import glob
import shutil

# goal: create ehr_ppg_cxr_dataset/train and ehr_ppg_cxr_dataset/test directories with the following files:
#       ehr_ppg_cxr_dataset/{partition}/{subject_id}_{stay_id}_timeseries.csv (one per episode)
#       ehr_ppg_cxr_dataset/{partition}/listfile.csv (one per partition)

def process_partition(args, partition, eps=1e-6):
    non_ts_train = []
    non_ts_test = []
    partition_df = pd.read_csv(args.ehr_ppg_cxr_partition)

    # iterate over all episode sub-dir. in ehr_root/{partition}
    episodes = os.listdir(os.path.join(args.episode_root_path, partition))
    for episode in tqdm(episodes, desc=f'Iterating over episodes in ehr_root/{partition}'):
        # check if this episode has a linked PPG dataset and CXR image
        episode_dir = os.path.join(args.episode_root_path, partition, episode)
        ppg_file = glob.glob(os.path.join(episode_dir, '*ppg.csv'))
        cxr_file = glob.glob(os.path.join(episode_dir, '*.jpg'))
        if len(ppg_file) < 1 or len(cxr_file) < 1:
            continue    # if episode has is no linked PPG dataset and CXR image, it is not in the EHR+PPG+CXR dataset

        # --- continue only if this episode is in the EHR+PPG+CXR dataset --- #
        # iterate over all episode{#}_timeseries.csv files in this episode sub-dir. (one exists per episode/ICU stay/admission)
        episode_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(episode_dir)))
        for ts_filename in episode_ts_files:
            with open(os.path.join(episode_dir, ts_filename)) as tsfile:
                # 1/8: get length of stay (los) of this episode in hours
                lb_filename = ts_filename.replace("_timeseries", "")    # corresponding non-timeseries episode{#}.csv file
                label_df = pd.read_csv(os.path.join(episode_dir, lb_filename))
                if label_df.shape[0] == 0:
                    continue    # exclude this episode if the corresponding non-timeseries file is empty
                los = 24.0 * label_df.iloc[0]['Length of Stay']
                if pd.isnull(los):
                    print("\n\t(Length of stay is missing)", episode, ts_filename)
                    continue    # exclude this episode if the los is missing
                
                # 2/8: keep only those rows of the timeseries file that were recorded during the episode/ICU stay 
                # read the timeseries file into a list of lines (row [0] : header, rows [1,:] : events)
                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines] # 'Hours' col of timeseries file as a list
                ts_lines = [line for (line, t) in zip(ts_lines, event_times) if -eps < t < los + eps]
                if len(ts_lines) == 0:
                    print("\n\t(No events during this episode/ICU stay) ", episode, ts_filename)
                    continue    # exclude this episode if no events were recorded during this episode
                
                # 3/8: identify the partition of the EHR+PPG+CXR dataset that this episode belongs to
                partition_subjects = partition_df['<subject_id>_<stay_id>']
                episode_match = partition_df[partition_subjects == episode].partition.values
                if episode_match[0] == 'train':
                    output_dir = os.path.join(args.output_path, "train")
                elif episode_match[0] == 'test':
                    output_dir = os.path.join(args.output_path, "test")
                
                # 4/8: write the modified timeseries file into a new csv file at the corresponding ehr_ppg_cxr_dataset/{partition}
                output_ts_filename = episode + "_" + ts_filename.split('_')[1]    # output_ts_filename = {subject_id}_{stay_id}_timeseries.csv
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                # 5/8: get the label associated with this episode
                t2dm_label = 0
                for col in label_df.columns:
                    if col.startswith('Diagnosis'):
                        if label_df[col].iloc[0] == 1:
                            t2dm_label = 1
                            break
                
                # 6/8: get the stay_id, gender, age, BMI and family history of T2DM associated with this episode
                stay_id = label_df.iloc[0]['Icustay']
                gender = label_df.iloc[0]['Gender']
                age = label_df.iloc[0]['Age']
                bmi = label_df.iloc[0]['BMI']
                family_history = label_df.iloc[0]['Family history']
                
                # 7/8: move this episode's linked CXR image to ehr_ppg_cxr_dataset/CXR_images and get its file path
                cxr_dir = os.path.join(args.output_path, 'CXR_images', episode)
                if not os.path.exists(cxr_dir):
                    os.makedirs(cxr_dir)
                shutil.copy(cxr_file[0], os.path.join(cxr_dir, os.path.basename(cxr_file[0])))
                cxr_image_path = os.path.join(cxr_dir, os.path.basename(cxr_file[0]))

                # 8/8: get the PPG data associated with this episode
                with open(ppg_file[0]) as ppgfile:
                    ppg_lines = ppgfile.readlines()
                    ppg_lines = [line.strip() for line in ppg_lines[1:]]  # skip header as it is 'ppg'

                # non_ts_{partition} is a list of tuples, with each tuple corresponding to one episode of a subject
                if episode_match == 'train':
                    non_ts_train.append((output_ts_filename, stay_id, t2dm_label, gender, age, bmi, family_history, ppg_lines, cxr_image_path))
                elif episode_match == 'test':
                    non_ts_test.append((output_ts_filename, stay_id, t2dm_label, gender, age, bmi, family_history, ppg_lines, cxr_image_path))

    return non_ts_train, non_ts_test

def main():
    parser = argparse.ArgumentParser(description="15_create_ehr_ppg_cxr_dataset.py")
    parser.add_argument('episode_root_path', type=str, help="'ehr_root' directory containing EHR train, val and test sets.")
    parser.add_argument('output_path', type=str, help="'ehr_ppg_cxr_dataset' directory where the created EHR+PPG+CXR dataset should be written.")
    parser.add_argument('--ehr_ppg_cxr_partition', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'ehr_ppg_cxr_partition.csv'),
                    help="CSV file containing the partition split of the EHR+PPG+CXR dataset.")
    args = parser.parse_args()

    samples_train_from_ehr_train, samples_test_from_ehr_train = process_partition(args, "train")
    samples_train_from_ehr_val, samples_test_from_ehr_val = process_partition(args, "val")
    samples_train_from_ehr_test, samples_test_from_ehr_test = process_partition(args, "test")

    # concatenate samples_{partition} obtained from iterating over episodes in all three partitions in ehr_root
    samples_train = samples_train_from_ehr_train + samples_train_from_ehr_val + samples_train_from_ehr_test
    samples_test = samples_test_from_ehr_train + samples_test_from_ehr_val + samples_test_from_ehr_test
    
    # save samples_{partition} as a single non-timeseries file, listfile.csv file, at ehr_ppg_cxr_dataset/{partition}
    random.shuffle(samples_train) # shuffle the samples in train set
    samples_test = sorted(samples_test)
    listfile_header = "timeseries,stay_id,label,gender,age,bmi,family_history,ppg,cxr_path"
    with open(os.path.join(args.output_path, "train", "listfile.csv"), "w") as listfile:
        listfile.write(listfile_header + "\n")
        for (x, stay_id, l, g, a, b, fam, ppg, cxr) in samples_train:
            listfile.write('{},{},{},{},{},{:.2f},{},{},{}\n'.format(x, stay_id, l, g, a, b, fam, ppg, cxr))
    with open(os.path.join(args.output_path, "test", "listfile.csv"), "w") as listfile:
        listfile.write(listfile_header + "\n")
        for (x, stay_id, l, g, a, b, fam, ppg, cxr) in samples_test:
            listfile.write('{},{},{},{},{},{:.2f},{},{},{}\n'.format(x, stay_id, l, g, a, b, fam, ppg, cxr))

if __name__ == '__main__':
    main()