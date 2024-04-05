import os
import numpy as np
from PIL import Image
import torch

from torch.utils.data import Dataset
from vilt.transforms import keys_to_transforms

class EHRCXRDataset(Dataset):
    def __init__(self, discretizer, normalizer, split, dataset_dir, max_text_len, transform_keys, image_size):
        self.discretizer = discretizer
        self.normalizer = normalizer
        assert split in ["train", "val", "test"]
        self.split = split
        self._dataset_dir = dataset_dir
        self.max_text_len = max_text_len    # max_text_len - 1 is passed since [class] token is prepended later
        self.transforms = keys_to_transforms(transform_keys, size=image_size)  # defined in vilt/transforms/pixelbert.py
        
        # read listfile.csv and store its data in self.data_map
        listfile_path = os.path.join(dataset_dir, split, "listfile.csv")
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()                      # list of strings, each a row in listfile.csv
        self._listfile_header = self._data[0]
        self._data = self._data[1:]
        self._data = [line.split(',') for line in self._data]   # split each row in listfile.csv (string in list) by comma, thus creating a list of lists of strings
        # self.data_map is a dict whose keys are timeseries filenames and values are dicts with remaining cols in listfile and their values in appropriate data types
        self.data_map = {
            mas[0]: {
                'stay_id': int(float(mas[1])),
                'label': int(mas[2]),
                'gender': int(float(mas[3])),
                'age': float(mas[4]),
                'family_history': int(float(mas[6])),
                'img_path': mas[7].split('\n')[0]
            } for mas in self._data
        }
        self.names = list(self.data_map.keys())     # list of timeseries filenames
    
    def read_timeseries(self, ts_filename):        
        ret = []
        with open(os.path.join(self._dataset_dir, self.split, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)  # ret is a list of np arrays, each a row in the timeseries file
    
    def read_by_ts_filename(self, index):
        if isinstance(index, int):
            index = self.names[index]
        (X, header) = self.read_timeseries(index)
        ret = self.data_map[index]
        ret.update({'ts_filename': index, 'ts_rows': X, 'ts_header': header})
        return ret
    
    def get_text(self, index):
        if isinstance(index, int):
            index = self.names[index]
        ret = self.read_by_ts_filename(index)
        ts_data = ret["ts_rows"]
        non_ts_data = {"gender": ret["gender"], "age": ret["age"], "family_history": ret["family_history"]}

        # discretized + normalised timeseries and non-ts EHR data of this sample and its mask
        data, mask = self.discretizer.transform(ts_data, non_ts_data, max_rows=self.max_text_len)
        data = self.normalizer.transform(data)

        label = int(ret["label"])
        img_path = ret["img_path"]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), label, img_path

    def get_image(self, img_path):
        assert isinstance(img_path, str)
        img = Image.open(img_path).convert('RGB')
        image_tensor = [tr(img) for tr in self.transforms]  # list of 3D tensors of transformed images of this sample (only one element since self.transforms_keys is only ["pixelbert"])
        return image_tensor

    def __getitem__(self, index):
        data, mask, label, img_path = self.get_text(index)
        image_tensor = self.get_image(img_path)
        return {
            "image": image_tensor,  # a list with a single 3D tensor (dim: 3 x H x W)
            "text": data,           # a 2D tensor (dim: no. of bins, no. of timeseries and non-ts variables)
            "mask": mask,           # a 1D tensor (dim: no. of bins)
            "label": label,         # a scalar (1: T2DM, 0: no T2DM)
        }

    def __len__(self):
        return len(self.names)

    def my_collate(self, batch):                                                    # batch = [{'image': 3D tensor, ..., 'label': scalar}, ..., {'image': 3D tensor, ..., 'label': scalar}]
        # RE-STRUCTURE BATCH #
        batch_size = len(batch)
        keys = set([key for sample in batch for key in sample.keys()])              # ('image', 'text', 'mask', 'label')
        dict_batch = {key: [sample[key] for sample in batch] for key in keys}       # {'image': [list of 3D tensors], 'text': [list of 2D tensors], 'mask': [list of 1D tensors], 'label': [list of scalars]}

        # IMAGE MODALITY - PAD SMALLER IMAGES #
        img = dict_batch["image"]
        img_sizes = list()
        img_sizes += [ii.shape for i in img if i is not None for ii in i]           # a list of 3-element tuples representing the dim of each transformed image in batch
        for size in img_sizes:
            assert (len(size) == 3), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"    # must be 3 since images are RGB

        # maximum height and width among all images in the batch (images' dim can vary since images underwent transformations)
        max_height = max([i[1] for i in img_sizes])
        max_width = max([i[2] for i in img_sizes])

        # resize smaller images to (max_height x max_width) by padding with zeros
        view_size = len(img[0])                     # no. of transformed image of each sample in batch (i.e., 1 if self.transforms_keys is only ["pixelbert"])
        new_images = [torch.zeros(batch_size, 3, max_height, max_width) for _ in range(view_size)]

        for bi in range(batch_size):
            orig_batch = img[bi]                    # a single 3D tensor of the transformed image of sample 'bi' in the batch
            for vi in range(view_size):             # one iteration since self.transforms_keys is only ["pixelbert"]
                if orig_batch is None:
                    new_images[vi][bi] = None
                else:
                    orig = img[bi][vi]
                    new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

        dict_batch["image"] = new_images            # dict_batch["image"] = list containing a single 4D tensor of shape: (batch_size, 3, max_height, max_width)

        # TEXT MODALITY - PAD EHR DATA AND MASK TENSORS #
        data = dict_batch["text"]
        data_np = [np.array(d) for d in data]       # convert to a list of 2D np arrays for padding operation
        data_padded = self.pad_zeros_data(data_np)  # list (len: batch_size) of 2D tensors (shape: max_text_len-1, ehr_n_var)
        data_tensor = torch.stack(data_padded)
        dict_batch["text"] = data_tensor            # dict_batch["text"] = 3D tensor of shape: (batch_size, max_text_len-1, ehr_n_var)

        mask = dict_batch["mask"]
        mask_np = [np.array(m) for m in mask]       # convert to a list of 1D np arrays for padding operation
        mask_padded = self.pad_zeros_mask(mask_np)  # list (len: batch_size) of 1D tensors (len: max_text_len-1)
        mask_tensor = torch.stack(mask_padded)
        dict_batch["mask"] = mask_tensor            # dict_batch["mask"] = 2D tensor with shape: (batch_size, max_text_len-1)

        # batch["label"] = list of scalars representing the labels of the samples in the batch (unchanged)
        
        return dict_batch

    def pad_zeros_data(self, data):
        dtype = data[0].dtype
        # concatenate each 2D data with a zero-filled array (of appropriate size) such that padded_data.shape[0] = max_text_len-1
        padded_data = [np.concatenate([x, np.zeros((self.max_text_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) for x in data]
        return [torch.tensor(padded) for padded in padded_data]     # convert to a list of 2D tensors
    
    def pad_zeros_mask(self, mask):
        dtype = mask[0].dtype
        # concatenate each mask 1D array with a zero-filled array (of appropriate size) such that padded_mask.shape[0] = max_text_len-1
        padded_mask = [np.concatenate([x, np.zeros(self.max_text_len - x.shape[0], dtype=dtype)]) for x in mask]
        return [torch.tensor(padded) for padded in padded_mask]     # convert to a list of 1D tensors