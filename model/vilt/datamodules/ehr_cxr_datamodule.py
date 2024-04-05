import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from vilt.datasets import EHRCXRDataset
from vilt.ehr_utils.utils import Discretizer, Normalizer

class EHRCXRDataModule(LightningDataModule):  # BaseDataModule is a PyTorch Lightning's LightningDataModule subclass
    def __init__(self, _config):
        super().__init__()

        self.data_dir = _config["data_root"]                # ./ehr_cxr_dataset; path to the dataset relative to run.py
        self.num_workers = _config["num_workers"]           # 8; number of subprocesses for data loading
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size
        self.image_size = _config["image_size"]             # 384
        self.max_text_len = _config["max_text_len"]-1       # self.max_text_len is -1 since [class] token is prepended later
        self.timestep = _config["timestep"]
        self.impute_strategy = _config["impute_strategy"]
        self.train_transform_keys = (                       # ['pixelbert']
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )
        self.val_transform_keys = (                         # ['pixelbert']
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        self.setup_flag = False                             # flag to check if setup() has been called

    @property
    def dataset_cls(self):  # a class property
        return EHRCXRDataset
    
    # initialise the "train" instance of the dataset class (defined in a vilt/datasets module) that is returned as the dataset_cls property using the following arguments
    def set_train_dataset(self):
        discretizer = Discretizer(timestep=self.timestep, impute_strategy=self.impute_strategy, store_masks=True, start_time='zero', config_path='vilt/ehr_utils/ehr_cxr_discretizer_config.json')
        normalizer = Normalizer()
        if self.impute_strategy == 'mean':
            normalizer_state = 'normalizer__{}_{}h_zero'.format(self.max_text_len, self.timestep)
        else:
            normalizer_state = 'normalizer__{}_{}h_{}'.format(self.max_text_len, self.timestep, self.impute_strategy)
        normalizer_path = os.path.join(os.path.dirname(__file__), "..", "ehr_utils", normalizer_state)
        normalizer.load_params(normalizer_path)
        self.train_dataset = self.dataset_cls(
            discretizer,
            normalizer,
            split="train",
            dataset_dir=self.data_dir,
            max_text_len=self.max_text_len,
            transform_keys=self.train_transform_keys,
            image_size=self.image_size
        )

    # initialise the "val" instance of the dataset class (defined in a vilt/datasets module) that is returned as the dataset_cls property using the following arguments
    def set_val_dataset(self):
        discretizer = Discretizer(timestep=self.timestep, impute_strategy=self.impute_strategy, store_masks=True, start_time='zero', config_path='vilt/ehr_utils/ehr_cxr_discretizer_config.json')
        normalizer = Normalizer()
        if self.impute_strategy == 'mean':
            normalizer_state = 'normalizer__{}_{}h_zero'.format(self.max_text_len, self.timestep)
        else:
            normalizer_state = 'normalizer__{}_{}h_{}'.format(self.max_text_len, self.timestep, self.impute_strategy)
        normalizer_path = os.path.join(os.path.dirname(__file__), "..", "ehr_utils", normalizer_state)
        normalizer.load_params(normalizer_path)
        self.val_dataset = self.dataset_cls(
            discretizer,
            normalizer,
            split="val",
            dataset_dir=self.data_dir,
            max_text_len=self.max_text_len,
            transform_keys=self.val_transform_keys,
            image_size=self.image_size
        )

    # initialise the "test" instance of the dataset class (defined in a vilt/datasets module) that is returned as the dataset_cls property using the following arguments
    def set_test_dataset(self):
        discretizer = Discretizer(timestep=self.timestep, impute_strategy=self.impute_strategy, store_masks=True, start_time='zero', config_path='vilt/ehr_utils/ehr_cxr_discretizer_config.json')
        normalizer = Normalizer()
        if self.impute_strategy == 'mean':
            normalizer_state = 'normalizer__{}_{}h_zero'.format(self.max_text_len, self.timestep)
        else:
            normalizer_state = 'normalizer__{}_{}h_{}'.format(self.max_text_len, self.timestep, self.impute_strategy)
        normalizer_path = os.path.join(os.path.dirname(__file__), "..", "ehr_utils", normalizer_state)
        normalizer.load_params(normalizer_path)
        self.test_dataset = self.dataset_cls(
            discretizer,
            normalizer,
            split="test",
            dataset_dir=self.data_dir,
            max_text_len=self.max_text_len,
            transform_keys=self.val_transform_keys,
            image_size=self.image_size
        )

    # prepare_data() is not overridden as distributed training is not used

    def setup(self, stage):
        if not self.setup_flag:      # only implemented if setup() has not been called already
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.setup_flag = True   # to indicate that setup() has been called once

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.my_collate,  # when DataLoader retrieves a list of samples, it passes them to collator, which combines them into batches
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.my_collate, # when DataLoader retrieves a list of samples, it passes them to collator, which combines them into batches
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.my_collate, # when DataLoader retrieves a list of samples, it passes them to collator, which combines them into batches
        )
        return loader
    
class EHRCXRDataModule_comp(LightningDataModule):  # BaseDataModule is a PyTorch Lightning's LightningDataModule subclass
    def __init__(self, _config):
        super().__init__()

        self.data_dir = _config["data_root"]                # ./ehr_cxr_dataset_comp; path to the dataset relative to run.py
        self.num_workers = _config["num_workers"]           # 8; number of subprocesses for data loading
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size
        self.image_size = _config["image_size"]             # 384
        self.max_text_len = _config["max_text_len"]-1       # self.max_text_len is -1 since [class] token is prepended later
        self.timestep = _config["timestep"]
        self.impute_strategy = _config["impute_strategy"]
        self.train_transform_keys = (                       # ['pixelbert']
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )
        self.val_transform_keys = (                         # ['pixelbert']
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        self.setup_flag = False                             # flag to check if setup() has been called

    @property
    def dataset_cls(self):  # a class property
        return EHRCXRDataset
    
    # initialise the "train" instance of the dataset class (defined in a vilt/datasets module) that is returned as the dataset_cls property using the following arguments
    def set_train_dataset(self):
        discretizer = Discretizer(timestep=self.timestep, impute_strategy=self.impute_strategy, store_masks=True, start_time='zero', config_path='vilt/ehr_utils/ehr_cxr_discretizer_config.json')
        normalizer = Normalizer()
        if self.impute_strategy == 'mean':
            normalizer_state = 'normalizer__{}_{}h_zero'.format(self.max_text_len, self.timestep)
        else:
            normalizer_state = 'normalizer__{}_{}h_{}'.format(self.max_text_len, self.timestep, self.impute_strategy)
        normalizer_path = os.path.join(os.path.dirname(__file__), "..", "ehr_utils", normalizer_state)
        normalizer.load_params(normalizer_path)
        self.train_dataset = self.dataset_cls(
            discretizer,
            normalizer,
            split="train",
            dataset_dir=self.data_dir,
            max_text_len=self.max_text_len,
            transform_keys=self.train_transform_keys,
            image_size=self.image_size
        )

    # initialise the "val" instance of the dataset class (defined in a vilt/datasets module) that is returned as the dataset_cls property using the following arguments
    def set_val_dataset(self):
        discretizer = Discretizer(timestep=self.timestep, impute_strategy=self.impute_strategy, store_masks=True, start_time='zero', config_path='vilt/ehr_utils/ehr_cxr_discretizer_config.json')
        normalizer = Normalizer()
        if self.impute_strategy == 'mean':
            normalizer_state = 'normalizer__{}_{}h_zero'.format(self.max_text_len, self.timestep)
        else:
            normalizer_state = 'normalizer__{}_{}h_{}'.format(self.max_text_len, self.timestep, self.impute_strategy)
        normalizer_path = os.path.join(os.path.dirname(__file__), "..", "ehr_utils", normalizer_state)
        normalizer.load_params(normalizer_path)
        self.val_dataset = self.dataset_cls(
            discretizer,
            normalizer,
            split="test",   ### modified from "val" to "test"
            dataset_dir=self.data_dir,
            max_text_len=self.max_text_len,
            transform_keys=self.val_transform_keys,
            image_size=self.image_size
        )

    # initialise the "test" instance of the dataset class (defined in a vilt/datasets module) that is returned as the dataset_cls property using the following arguments
    def set_test_dataset(self):
        discretizer = Discretizer(timestep=self.timestep, impute_strategy=self.impute_strategy, store_masks=True, start_time='zero', config_path='vilt/ehr_utils/ehr_cxr_discretizer_config.json')
        normalizer = Normalizer()
        if self.impute_strategy == 'mean':
            normalizer_state = 'normalizer__{}_{}h_zero'.format(self.max_text_len, self.timestep)
        else:
            normalizer_state = 'normalizer__{}_{}h_{}'.format(self.max_text_len, self.timestep, self.impute_strategy)
        normalizer_path = os.path.join(os.path.dirname(__file__), "..", "ehr_utils", normalizer_state)
        normalizer.load_params(normalizer_path)
        self.test_dataset = self.dataset_cls(
            discretizer,
            normalizer,
            split="test",
            dataset_dir=self.data_dir,
            max_text_len=self.max_text_len,
            transform_keys=self.val_transform_keys,
            image_size=self.image_size
        )

    # prepare_data() is not overridden as distributed training is not used

    def setup(self, stage):
        if not self.setup_flag:      # only implemented if setup() has not been called already
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.setup_flag = True   # to indicate that setup() has been called once

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.my_collate,  # when DataLoader retrieves a list of samples, it passes them to collator, which combines them into batches
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.my_collate, # when DataLoader retrieves a list of samples, it passes them to collator, which combines them into batches
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.my_collate, # when DataLoader retrieves a list of samples, it passes them to collator, which combines them into batches
        )
        return loader