import os
import random
import pandas as pd
import shutil
import argparse
from tqdm import tqdm

# goal: [1] move the timeseries and non-timeseries files of the subjects in 'root' to separate episode sub-dir. in 'ehr_root'
#       [2] split the episode sub-dir. into train (70%), dev (10%) and test (20%) sets
#       [3] move the episode sub-dir. to their corresponding partition, i.e. 'ehr_root/train', 'ehr_root/dev' or 'ehr_root/test'

def re_group_episodes(subjects_root_path, episodes_root_path):
    os.makedirs(episodes_root_path, exist_ok=True)
    # iterate over all subject sub-dir. in root dir
    subjects = list(filter(str.isdigit, os.listdir(subjects_root_path)))
    for subject_id in tqdm(subjects, desc=f'Moving timeseries and non-ts files to {episodes_root_path}'):
        episode_files = {}  # dict. for pairs of #: [episode{#}_timeseries.csv, episode{#}.csv]
        # iterate over all files in the subject sub-dir.
        for file in os.listdir(os.path.join(subjects_root_path, subject_id)):
            if "episode" in file:   # identify the timeseries (episode{#}_timeseries.csv) and non-timeseries files (episode{#}.csv)
                # extract episode number # from the filename
                if "timeseries" in file:
                    episode_number = file.split('_')[0].replace('episode', '')
                else:
                    episode_number = file.split('.')[0].replace('episode', '')
                # create a key-value pair in episode_files for each unique #
                if episode_number not in episode_files:
                    episode_files[episode_number] = []
                # append the timeseries or non-timeseries files to the corresponding #
                episode_files[episode_number].append(file)

        for episode_number, files in episode_files.items(): # iterate over all key-value pairs in episode_files
            for file in files:  # iterate over the timeseries and non-timeseries file of each episode
                # extract the stay_id from the non-timeseries file
                if "timeseries" not in file:
                    df = pd.read_csv(os.path.join(subjects_root_path, subject_id, file))
                    if df.shape[0] > 0:
                        stay_id = df["Icustay"].iloc[0]
                    else:
                        break   # exclude this episode from the EHR datatset if its non-timeseries file is empty
                # move the two files of each episode to ehr_root/{subject_id}_{stay_id}
                if not os.path.exists(os.path.join(episodes_root_path, f'{subject_id}_{stay_id}')):
                    os.mkdir(os.path.join(episodes_root_path, f'{subject_id}_{stay_id}'))
                shutil.move(os.path.join(subjects_root_path, subject_id, file), os.path.join(episodes_root_path, f'{subject_id}_{stay_id}', file))        
                # following this point, subject sub-dir. in 'root' do not contain any timeseries or non-timeseries files

def partition_split(episodes_root_path, train_ratio=0.7, dev_ratio=0.1):
    random.seed(0)
    episodes = os.listdir(episodes_root_path)
    random.shuffle(episodes)
    train_episodes = episodes[:int(len(episodes)*train_ratio)]
    dev_episodes = episodes[int(len(episodes)*train_ratio):int(len(episodes)*(train_ratio+dev_ratio))]
    test_episodes = episodes[int(len(episodes)*(train_ratio+dev_ratio)):]

    assert len(set(train_episodes) & set(dev_episodes)) == 0
    assert len(set(train_episodes) & set(test_episodes)) == 0
    assert len(set(dev_episodes) & set(test_episodes)) == 0

    # create a df (+ save as csv) indicating the partition that each episode belongs to
    partition_list = [(episode, 'train') for episode in train_episodes] + \
                     [(episode, 'dev') for episode in dev_episodes] + \
                     [(episode, 'test') for episode in test_episodes]
    partition_df = pd.DataFrame(partition_list, columns=['<subject_id>_<stay_id>', 'partition'])
    partition_df.to_csv(os.path.join(os.path.dirname(__file__), 'ehr_partition.csv'), index=False)

def move_to_partition(episodes_root_path, episodes, partition):
    if not os.path.exists(os.path.join(episodes_root_path, partition)):
        os.mkdir(os.path.join(episodes_root_path, partition))
    for episode in episodes:
        src = os.path.join(episodes_root_path, episode)
        dest = os.path.join(episodes_root_path, partition, episode)
        shutil.move(src, dest)  # move episode sub-dir. from ehr_root to ehr_root/{partition}
    
def main():
    parser = argparse.ArgumentParser(description="4_partition_ehr_samples.py")
    parser.add_argument('subjects_root_path', type=str, help="'root' directory containing subject sub-dir.")
    parser.add_argument('episodes_root_path', type=str, help="'ehr_root' directory where episode sub-dir are to be stored.")
    args = parser.parse_args()

    re_group_episodes(args.subjects_root_path, args.episodes_root_path)
    partition_split(args.episodes_root_path, train_ratio=0.7, dev_ratio=0.1)

    train_set = set()
    dev_set = set()
    test_set = set()
    with open(os.path.join(os.path.dirname(__file__), 'ehr_partition.csv'), "r") as partition_file:
        next(partition_file)   # skip header
        for line in partition_file:
            x, y = line.strip().split(',')
            if str(y) == "train":
                train_set.add(x)
            elif str(y) == "dev":
                dev_set.add(x)
            elif str(y) == "test":
                test_set.add(x)

    episodes = os.listdir(args.episodes_root_path)
    train_episodes = [x for x in episodes if x in train_set]
    dev_episodes = [x for x in episodes if x in dev_set]
    test_episodes = [x for x in episodes if x in test_set]
    
    assert len(set(train_episodes) & set(dev_episodes)) == 0
    assert len(set(train_episodes) & set(test_episodes)) == 0
    assert len(set(dev_episodes) & set(test_episodes)) == 0

    move_to_partition(args.episodes_root_path, train_episodes, "train")
    move_to_partition(args.episodes_root_path, dev_episodes, "dev")
    move_to_partition(args.episodes_root_path, test_episodes, "test")

if __name__ == '__main__':
    main()