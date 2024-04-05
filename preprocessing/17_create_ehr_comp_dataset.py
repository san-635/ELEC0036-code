import os
import argparse
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm
import glob
import shutil

# goal: create ehr_dataset_comp/train and ehr_dataset_comp/test directories with the following files:
#       ehr_dataset_comp/{partition}/{subject_id}_{stay_id}_timeseries.csv (one per episode)
#       ehr_dataset_comp/{partition}/listfile.csv (one per partition)

def process_partition(args, partition, eps=1e-6):
    non_ts_train = []
    non_ts_test = []
    partition_df = pd.read_csv(args.compare_partition)

    # iterate over all episodes listed in ehr_ppg_cxr_partition.csv
    for _, row in tqdm(partition_df.iterrows(), total=partition_df.shape[0], desc=f'Iterating over episodes in ehr_root/{partition}'):
        episode = row['<subject_id>_<stay_id>']
        episode_match = row['partition']

        # check if this episode exists in ehr_root/{partition} and store its path
        episode_dir = os.path.join(args.episode_root_path, partition, episode)
        if not os.path.exists(episode_dir):
            continue
        output_dir = os.path.join(args.output_path, episode_match)        

        # --- continue only if this episode is also in the EHR+PPG+CXR dataset --- #
        # iterate over all episode{#}_timeseries.csv files in this episode sub-dir. (one exists per episode/ICU stay/admission)
        episode_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(episode_dir)))
        for ts_filename in episode_ts_files:
            with open(os.path.join(episode_dir, ts_filename)) as tsfile:
                # 1/6: get length of stay (los) of this episode in hours
                lb_filename = ts_filename.replace("_timeseries", "")    # corresponding non-timeseries episode{#}.csv file
                label_df = pd.read_csv(os.path.join(episode_dir, lb_filename))
                if label_df.shape[0] == 0:
                    continue    # exclude this episode if the corresponding non-timeseries file is empty
                los = 24.0 * label_df.iloc[0]['Length of Stay']
                if pd.isnull(los):
                    print("\n\t(Length of stay is missing)", episode, ts_filename)
                    continue    # exclude this episode if the los is missing
                
                # 2/6: keep only those rows of the timeseries file that were recorded during the episode/ICU stay 
                # read the timeseries file into a list of lines (row [0] : header, rows [1,:] : events)
                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines] # 'Hours' col of timeseries file as a list
                ts_lines = [line for (line, t) in zip(ts_lines, event_times) if -eps < t < los + eps]
                if len(ts_lines) == 0:
                    print("\n\t(No events during this episode/ICU stay) ", episode, ts_filename)
                    continue    # exclude this episode if no events were recorded during this episode
                
                # 3/6: write the modified timeseries file into a new csv file at the corresponding ehr_dataset_comp/{partition}
                output_ts_filename = episode + "_" + ts_filename.split('_')[1]    # output_ts_filename = {subject_id}_{stay_id}_timeseries.csv
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                # 4/6: get the label associated with this episode
                t2dm_label = 0
                for col in label_df.columns:
                    if col.startswith('Diagnosis'):
                        if label_df[col].iloc[0] == 1:
                            t2dm_label = 1
                            break
                
                # 5/6: get the stay_id, gender, age, BMI and family history of T2DM associated with this episode
                stay_id = label_df.iloc[0]['Icustay']
                gender = label_df.iloc[0]['Gender']
                age = label_df.iloc[0]['Age']
                bmi = label_df.iloc[0]['BMI']
                family_history = label_df.iloc[0]['Family history']

                # non_ts_{partition} is a list of tuples, with each tuple corresponding to one episode of a subject
                if episode_match == 'train':
                    non_ts_train.append((output_ts_filename, stay_id, t2dm_label, gender, age, bmi, family_history))
                elif episode_match == 'test':
                    non_ts_test.append((output_ts_filename, stay_id, t2dm_label, gender, age, bmi, family_history))

    return non_ts_train, non_ts_test

def main():
    parser = argparse.ArgumentParser(description="17_create_ehr_comp_dataset.py")
    parser.add_argument('episode_root_path', type=str, help="'ehr_root' directory containing EHR train, val and test sets.")
    parser.add_argument('output_path', type=str, help="'ehr_dataset_comp' directory where the created EHR dataset should be written.")
    parser.add_argument('--compare_partition', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'ehr_ppg_cxr_partition.csv'),
                    help="CSV file containing the partition split of the EHR+PPG+CXR dataset.")
    args = parser.parse_args()

    samples_train_from_ehr_train, samples_test_from_ehr_train = process_partition(args, "train")
    samples_train_from_ehr_val, samples_test_from_ehr_val = process_partition(args, "val")
    samples_train_from_ehr_test, samples_test_from_ehr_test = process_partition(args, "test")

    # concatenate samples_{partition} obtained from iterating over episodes in all three partitions in ehr_root
    samples_train = samples_train_from_ehr_train + samples_train_from_ehr_val + samples_train_from_ehr_test
    samples_test = samples_test_from_ehr_train + samples_test_from_ehr_val + samples_test_from_ehr_test
    
    # save samples_{partition} as a single non-timeseries file, listfile.csv file, at ehr_dataset_comp/{partition}
    random.shuffle(samples_train) # shuffle the samples in train set
    samples_test = sorted(samples_test)
    listfile_header = "timeseries,stay_id,label,gender,age,bmi,family_history"
    with open(os.path.join(args.output_path, "train", "listfile.csv"), "w") as listfile:
        listfile.write(listfile_header + "\n")
        for (x, stay_id, l, g, a, b, fam) in samples_train:
            listfile.write('{},{},{},{},{},{:.2f},{}\n'.format(x, stay_id, l, g, a, b, fam))
    with open(os.path.join(args.output_path, "test", "listfile.csv"), "w") as listfile:
        listfile.write(listfile_header + "\n")
        for (x, stay_id, l, g, a, b, fam) in samples_test:
            listfile.write('{},{},{},{},{},{:.2f},{}\n'.format(x, stay_id, l, g, a, b, fam))

if __name__ == '__main__':
    main()