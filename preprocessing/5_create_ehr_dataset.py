import os
import argparse
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm

# goal: create ehr_dataset/train, ehr_dataset/dev and ehr_dataset/test directories with the following files:
#       ehr_dataset/{partition}/{subject_id}_{stay_id}_timeseries.csv (one per episode)
#       ehr_dataset/{partition}/listfile.csv (one per partition)

def process_partition(args, partition, eps=1e-6):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    non_ts = []
    # iterate over all episode sub-dir. in ehr_root/{partition}
    episodes = os.listdir(os.path.join(args.episode_root_path, partition))
    for episode in tqdm(episodes, desc=f'Iterating over episodes in ehr_root/{partition}'):
        episode_folder = os.path.join(args.episode_root_path, partition, episode)
        # iterate over all episode{#}_timeseries.csv files in this episode sub-dir. (only one exists per episode/ICU stay/admission)
        episode_ts_file = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(episode_folder)))
        for ts_filename in episode_ts_file:
            with open(os.path.join(episode_folder, ts_filename)) as tsfile:
                # 1/5: get length of stay (los) of this episode in hours
                lb_filename = ts_filename.replace("_timeseries", "")    # corresponding non-timeseries file (episode{#}.csv)
                label_df = pd.read_csv(os.path.join(episode_folder, lb_filename))
                if label_df.shape[0] == 0:
                    continue    # exclude this episode if the corresponding non-timeseries file is empty
                los = 24.0 * label_df.iloc[0]['Length of Stay']
                if pd.isnull(los):
                    print("\n\t(Length of stay is missing)", episode, ts_filename)
                    continue    # exclude this episode if the los is missing
                
                # 2/5: keep only those rows of the timeseries file that were recorded during the episode/ICU stay 
                # read the timeseries file into a list of lines (row [0] - header, rows [1,:] - events)
                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines] # 'Hours' col of timeseries file as a list
                ts_lines = [line for (line, t) in zip(ts_lines, event_times) if -eps < t < los + eps]
                if len(ts_lines) == 0:
                    print("\n\t(No events during this episode/ICU stay) ", episode, ts_filename)
                    continue    # exclude this episode if no events were recorded during this episode
                
                # 3/5: write the modified timeseries file of this episode into a new csv file at ehr_dataset/{partition}
                output_ts_filename = episode + "_" + ts_filename.split("_")[1]    # output_ts_filename = {subject_id}_{stay_id}_timeseries.csv
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                # 4/5: get the label associated with this episode
                t2dm_label = 0
                for col in label_df.columns:
                    if col.startswith('Diagnosis'):
                        if label_df[col].iloc[0] == 1:
                            t2dm_label = 1
                            break
                
                # 5/5: get the stay_id, gender, age, BMI and family history of T2DM associated with this episode
                stay_id = label_df.iloc[0]['Icustay']
                gender = label_df.iloc[0]['Gender']
                age = label_df.iloc[0]['Age']
                bmi = label_df.iloc[0]['BMI']
                family_history = label_df.iloc[0]['Family history']
                
                # non_ts is a list of tuples, with each tuple corresponding to one episode of a subject
                non_ts.append((output_ts_filename, stay_id, t2dm_label, gender, age, bmi, family_history))

    print(f"A total of {len(non_ts)} EHR samples exist in {partition}")
    if partition == "train":
        random.shuffle(non_ts) # shuffle the samples in train set
    if partition == "dev":
        non_ts = sorted(non_ts)
    if partition == "test":
        non_ts = sorted(non_ts)

    # save non_ts as a single non-timeseries file, listfile.csv file, for this partition at ehr_dataset/{partition}
    listfile_header = "timeseries,stay_id,label,gender,age,bmi,family_history"
    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write(listfile_header + "\n")
        for (x, stay_id, l, g, a, b, fam) in non_ts:
            listfile.write('{},{},{},{},{},{:.2f},{}\n'.format(x, stay_id, l, g, a, b, fam))

def main():
    parser = argparse.ArgumentParser(description="5_create_ehr_dataset.py")
    parser.add_argument('episode_root_path', type=str, help="'ehr_root' directory containing EHR train, dev and test sets.")
    parser.add_argument('output_path', type=str, help="'ehr_dataset' directory where the created EHR datasets should be stored.")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "train")
    process_partition(args, "dev")
    process_partition(args, "test")

if __name__ == '__main__':
    main()