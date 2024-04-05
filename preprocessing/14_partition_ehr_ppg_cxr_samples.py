import os
import random
import pandas as pd
import argparse
import glob
from tqdm import tqdm

# goal: split the episodes in 'ehr_root' with a linked PPG dataset and CXR image into train (80%) and test (20%) sets

def get_subdir_from_partition(episodes_root_path, partition):
    episodes = os.listdir(os.path.join(episodes_root_path, partition))

    # iterate over all episode sub-dir. in ehr_root and select those with a linked PPG dataset and CXR image
    ep_dir = []
    # for episode in episodes:
    for episode in tqdm(episodes, desc=f'Iterating over episodes in ehr_root/{partition}'):
        ppg_files = glob.glob(os.path.join(episodes_root_path, partition, episode, 'ppg.csv'))
        cxr_files = glob.glob(os.path.join(episodes_root_path, partition, episode, '*.jpg'))
        if ppg_files and cxr_files:
            ep_dir.append(episode)

    return ep_dir

def partition_split(ep_dir, train_ratio=0.8):
    random.seed(0)
    random.shuffle(ep_dir)
    train_episodes = ep_dir[:int(len(ep_dir)*train_ratio)]
    test_episodes = ep_dir[int(len(ep_dir)*(train_ratio)):]

    assert len(set(train_episodes) & set(test_episodes)) == 0

    # create a df (+ save as csv) indicating the partition that each episode belongs to
    partition_list = [(episode, 'train') for episode in train_episodes] + \
                     [(episode, 'test') for episode in test_episodes]
    partition_df = pd.DataFrame(partition_list, columns=['<subject_id>_<stay_id>', 'partition'])
    partition_df.to_csv(os.path.join(os.path.dirname(__file__), 'ehr_ppg_cxr_partition.csv'), index=False)

    # create a df (+ save as csv) of unique subjects in the EHR+PPG+CXR dataset
    subjects = list(set([episode.split('_')[0] for episode in ep_dir]))
    subjects_df = pd.DataFrame(subjects, columns=['<subject_id>'])
    subjects_df.to_csv(os.path.join(os.path.dirname(__file__), 'ehr_ppg_cxr_subjects.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description="14_partition_ehr_ppg_cxr_samples.py")
    parser.add_argument('episodes_root_path', type=str, help="'ehr_root' directory containing EHR train, val and test sets.")
    args = parser.parse_args()

    ep_dir_train = get_subdir_from_partition(args.episodes_root_path, "train")
    ep_dir_val = get_subdir_from_partition(args.episodes_root_path, "val")
    ep_dir_test = get_subdir_from_partition(args.episodes_root_path, "test")
    ep_dir = ep_dir_train + ep_dir_val + ep_dir_test
    
    partition_split(ep_dir, train_ratio=0.8)

if __name__ == '__main__':
    main()