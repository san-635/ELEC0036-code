import os
import random
import pandas as pd
import argparse
import glob

# goal: split the episodes in 'ehr_root' with linked PPG datasets into train (70%), val (10%) and test (20%) sets

def get_subdir_from_partition(episodes_root_path, partition):
    episodes = os.listdir(os.path.join(episodes_root_path, partition))

    # iterate over all episode sub-dir. in ehr_root and select those with a linked PPG dataset
    ppg_dir = []
    for episode in episodes:
        ppg_files = glob.glob(os.path.join(episodes_root_path, partition, episode, 'ppg.csv'))
        if ppg_files:
            ppg_dir.append(episode)

    return ppg_dir

def partition_split(ppg_dir, train_ratio=0.7, val_ratio=0.1):
    random.seed(0)
    random.shuffle(ppg_dir)
    train_episodes = ppg_dir[:int(len(ppg_dir)*train_ratio)]
    val_episodes = ppg_dir[int(len(ppg_dir)*train_ratio):int(len(ppg_dir)*(train_ratio+val_ratio))]
    test_episodes = ppg_dir[int(len(ppg_dir)*(train_ratio+val_ratio)):]

    assert len(set(train_episodes) & set(val_episodes)) == 0
    assert len(set(train_episodes) & set(test_episodes)) == 0
    assert len(set(val_episodes) & set(test_episodes)) == 0

    # create a df (+ save as csv) indicating the partition that each episode belongs to
    partition_list = [(episode, 'train') for episode in train_episodes] + \
                     [(episode, 'val') for episode in val_episodes] + \
                     [(episode, 'test') for episode in test_episodes]
    partition_df = pd.DataFrame(partition_list, columns=['<subject_id>_<stay_id>', 'partition'])
    partition_df.to_csv(os.path.join(os.path.dirname(__file__), 'ehr_ppg_partition.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description="10_partition_ehr_ppg_samples.py")
    parser.add_argument('episodes_root_path', type=str, help="'ehr_root' directory containing EHR train, val and test sets.")
    args = parser.parse_args()

    ppg_dir_train = get_subdir_from_partition(args.episodes_root_path, "train")
    ppg_dir_val = get_subdir_from_partition(args.episodes_root_path, "val")
    ppg_dir_test = get_subdir_from_partition(args.episodes_root_path, "test")
    ppg_dir = ppg_dir_train + ppg_dir_val + ppg_dir_test
    
    partition_split(ppg_dir, train_ratio=0.7, val_ratio=0.1)

if __name__ == '__main__':
    main()