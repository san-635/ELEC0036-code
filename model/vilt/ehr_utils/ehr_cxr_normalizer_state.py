from __future__ import absolute_import
from __future__ import print_function

from vilt.ehr_utils.utils import Discretizer, Normalizer
from vilt.datasets import EHRCXRDataset
from vilt.config import ex

import os
import copy

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    
    data_dir = _config['data_root']                # ./EHR_CXR_dataset
    image_size = _config["image_size"]             # 384
    max_text_len = _config["max_text_len"]-1       # -1 since [class] token is prepended later
    timestep = _config["timestep"]
    impute = _config["impute_strategy"]
    train_transform_keys = (                       # ['pixelbert']
        ["default_train"]
        if len(_config["train_transform_keys"]) == 0
        else _config["train_transform_keys"]
    )

    # create the discretizer
    discretizer = Discretizer(timestep=timestep, impute_strategy=impute, store_masks=True, start_time='zero', config_path='vilt/ehr_utils/ehr_cxr_discretizer_config.json')

    # create the normalizer
    normalizer = Normalizer()
    
    # create the train dataset
    reader = EHRCXRDataset(
        discretizer,
        normalizer,
        split="train",
        dataset_dir=data_dir,
        max_text_len=max_text_len,
        transform_keys=train_transform_keys,
        image_size=image_size
    )

    # read all samples in the train dataset and store the normalizer state in vilt/ehr_utils as a pickle file
    n_samples = len(reader)

    # read and discretize all samples in the train dataset
    for sample in range(n_samples):
        if sample % 1000 == 0:
            print('Processed {} / {} train dataset samples'.format(sample, n_samples), end='\r')
        ret = reader.read_by_ts_filename(sample)
        ts_data = ret["ts_rows"]
        # non_ts_data = {"gender": ret["gender"], "age": ret["age"], "bmi": ret["bmi"], "family_history": ret["family_history"]}
        non_ts_data = {"gender": ret["gender"], "age": ret["age"], "family_history": ret["family_history"]}
        data = discretizer.transform(ts_data, non_ts_data, max_text_len)[0]
        normalizer._feed_data(data)

    # save the normalizer state in vilt/ehr_utils as a pickle file
    file_name = 'normalizer__{}_{}h_{}'.format(max_text_len, timestep, impute)
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    print('Saving normalizer state to {} ...'.format(file_path))
    normalizer._save_params(file_path)