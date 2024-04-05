from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import platform
import pickle
import json
import os

class Discretizer:
    def __init__(self, timestep=0.25, impute_strategy='zero', store_masks=True, start_time='zero', config_path='vilt/ehr_utils/ehr_cxr_discretizer_config.json'):
        # READ JSON CONFIG FILE AND INITIALISATIONS #
        with open(config_path) as f:
            config_json = json.load(f)
            self._ts_variable = config_json['ts_variables']
            self._ts_variable_to_id = dict(zip(self._ts_variable, range(len(self._ts_variable))))   # i.e., {'Diastolic blood pressure': 0, 'Glucose': 1, ...}
            self._non_ts_variable = config_json['non_ts_variables']
            self._normal_values = config_json['normal_values']
        
        self._non_ts_variable_to_id = {}    # i.e., {'gender': 13, 'age': 14, 'family_history': 15}
        id = len(self._ts_variable)
        for non_ts_variable in self._non_ts_variable:
            self._non_ts_variable_to_id[non_ts_variable] = id
            id += 1
        
        self._header = ["Hours"] + self._ts_variable    # i.e., ['Hours', 'Diastolic blood pressure', 'Glucose', ..., 'Weight']
        self._timestep = timestep                       # config['timestep'] = 0.5
        self._store_masks = store_masks                 # True
        self._impute_strategy = impute_strategy         # config['impute_strategy'] = 'zero'
        self._start_time = start_time                   # 'zero'
        
        # STATS #
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

    def transform(self, ts_rows, non_ts_data, max_rows=None):
        header = self._header                           # only Hours and timeseries variable cols
        assert header[0] == "Hours"
        eps = 1e-6

        # NUMBER OF VARIABLES (COLS) #
        N_variables = len(self._ts_variable) + len(self._non_ts_variable)     # total number of timeseries and non-timeseries variables in EHR

        # ENSURE THAT ROWS ARE CHRONOLOGICALLY SORTED #
        ts = [float(row[0]) for row in ts_rows]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        # NUMBER OF BINS (ROWS) #
        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError('start_time is invalid')
        max_hours = max(ts) - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)    # + 1 to ensure that the last bin is not missed by int()
        if max_rows is not None:
            if N_bins > max_rows:   # if the number of bins is greater than max_text_len-1, it is capped and max_hours is recalculated
                N_bins = max_rows
                max_hours = N_bins * self._timestep - eps
            # else, N_bins is unchanged (can be smaller than max_text_len-1, shall be padded later with 0s)

        # this part is actually not necessary as there are no categorical variables in the timeseries files (i.e., each variable only occupies one col)
        cur_len = 0
        begin_pos = [0 for i in range(N_variables)]
        end_pos = [0 for i in range(N_variables)]
        for i in range(N_variables):
            begin_pos[i] = cur_len
            end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]    # cur_len is same as N_variables

        # DATA AND MASK #
        data = np.zeros(shape=(N_bins, N_variables), dtype=float)                       # shape: (<=max_text_len-1, ehr_n_var)
        mask = np.zeros(shape=(N_bins, N_variables), dtype=int)                         # shape: (<=max_text_len-1, ehr_n_var)
        original_value = [["" for j in range(N_variables)] for i in range(N_bins)]      # stores the original value -> for imputation

        # STATS #
        total_data = 0
        unused_data = 0

        # METHOD TO WRITE A VALUE TO A TIMESERIES VARIABLE IN A BIN #
        def write(data, bin_id, variable, value, begin_pos):
            variable_id = self._ts_variable_to_id[variable]
            data[bin_id, begin_pos[variable_id]] = float(value)  # assign the value to the corresponding bin

        # DISCRETIZATION OF TIMESERIES VARIABLE COLS #
        for row in ts_rows:                         # iterate over all rows of timeseries file
            # determine the bin to which this row belongs to based on the time in "Hours" col
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue                            # ignore rows in file that would be assigned to a bin that is larger than max_text_len-1
            bin_id = int(t / self._timestep - eps)
            if bin_id < 0 or bin_id >= N_bins:
                continue                            # ignore rows in file that would be assigned to a bin that is outside the range of bins

            # write/overwrite values to the corresponding timeseries variables of this bin
            for j in range(1, len(row)):            # iterate over all variable cols in this row
                if row[j] == "":
                    continue                        # ignore if value in col is NaN
                if j < len(header):
                    variable = header[j]            # if col is a timeseries variable
                else:
                    continue                        # ignore if col is a non-timeseries variable
                variable_id = self._ts_variable_to_id[variable]

                total_data += 1
                if mask[bin_id][variable_id] == 1:  # if mask is already 1 for this bin of this variable, update unused_data because the current value is overwritten
                    unused_data += 1
                mask[bin_id][variable_id] = 1       # indicate that a value is already present in this bin of this variable

                write(data, bin_id, variable, row[j], begin_pos)    # pass 2D np array, bin_id, timeseries variable name and its value, begin_pos
                original_value[bin_id][variable_id] = row[j]

        # IMPUTATION FOR TIMESERIES VARIABLE COLS #
        if self._impute_strategy not in ['zero', 'normal_value', 'mean', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._ts_variable))]
            for bin_id in range(N_bins):            # iterate over all bins
                for variable in self._ts_variable:  # iterate over all timeseries variables
                    variable_id = self._ts_variable_to_id[variable]
                    if mask[bin_id][variable_id] == 1:
                        prev_values[variable_id].append(original_value[bin_id][variable_id])
                        continue
                    # if mask[bin_id][variable_id] == 0, impute the value
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[variable]       # normal value
                    if self._impute_strategy == 'previous':
                        if len(prev_values[variable_id]) == 0:
                            imputed_value = self._normal_values[variable]   # normal value if no previous value
                        else:
                            imputed_value = prev_values[variable_id][-1]    # most recent value
                    write(data, bin_id, variable, imputed_value, begin_pos)

        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._ts_variable))]
            for bin_id in range(N_bins-1, -1, -1):   # iterate over all bins in reverse order
                for variable in self._ts_variable:   # iterate over all timeseries variables
                    variable_id = self._ts_variable_to_id[variable]
                    if mask[bin_id][variable_id] == 1:
                        prev_values[variable_id].append(original_value[bin_id][variable_id])
                        continue
                    # if mask[bin_id][variable_id] == 0, impute the value
                    if len(prev_values[variable_id]) == 0:
                        imputed_value = self._normal_values[variable]
                    else:
                        imputed_value = prev_values[variable_id][-1]
                    write(data, bin_id, variable, imputed_value, begin_pos)

        if self._impute_strategy == 'mean':
            load_file_path = os.path.join(os.path.dirname(__file__), 'normalizer__{}_{}h_zero'.format(max_rows, self._timestep))
            with open(load_file_path, "rb") as load_file:
                if platform.python_version()[0] == '2':
                    dict = pickle.load(load_file)
                else:
                    dict = pickle.load(load_file, encoding='latin1')
                col_means = dict['means']
            for bin_id in range(N_bins):            # iterate over all bins
                for variable in self._ts_variable:  # iterate over all timeseries variables
                    variable_id = self._ts_variable_to_id[variable]
                    if mask[bin_id][variable_id] == 1:
                        continue
                    write(data, bin_id, variable, col_means[variable_id], begin_pos)

        # STATS #
        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        # CONCATENATE NON-TIMESERIES VARIABLE COLS #
        for non_ts, value in non_ts_data.items():
            non_ts_id = self._non_ts_variable_to_id[non_ts]
            data[:, non_ts_id] = value
            mask[:, non_ts_id] = 1

        # indicates whether a bin has any non-zero data for any timeseries variable cols (ie, should model ignore this bin since it has no useful data apart from the uniform non-timeseries variables)
        mask_1d = np.any(mask[:, :len(self._ts_variable)], axis=1).astype(int)  # len: <=max_text_len-1

        if self._store_masks:
            return (data, mask_1d)          # return the discretized timeseries and non-timeseries data (<=max_text_len-1, ehr_n_var) and its mask (<=max_text_len-1)
        else:
            return data                     # return the discretized timeseries and non-timeseries data (<=max_text_len-1, ehr_n_var)

    def print_statistics(self):
        print("statistics of discretizer:")
        print("\tconverted {} examples".format(self._done_count))
        print("\taverage unused data = {:.2f} percent".format(100.0 * self._unused_data_sum / self._done_count))
        print("\taverage empty  bins = {:.2f} percent".format(100.0 * self._empty_bins_sum / self._done_count))

class Normalizer:
    def __init__(self):
        self._means = None
        self._stds = None
        
        self._sum_col = None
        self._sum_sq_col = None
        self._count = 0

    # calculate the running totals of count, sum and sum of squares of the cols over all samples in training set
    def _feed_data(self, data):
        data = np.array(data)
        self._count += data.shape[0]                     # running count of number of rows in each sample's data; scalar
        if self._sum_col is None:
            self._sum_col = np.sum(data, axis=0)
            self._sum_sq_col = np.sum(data**2, axis=0)
        else:
            self._sum_col += np.sum(data, axis=0)        # running sum of the cols of sample's data; 1D array of length ehr_n_var
            self._sum_sq_col += np.sum(data**2, axis=0)  # running sum of squares of the cols of sample's data; 1D array of length ehr_n_var

    # calculate the means and std devs of cols considering all samples in training set and save them to a pickle file
    def _save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_col
            self._stds = np.sqrt(1.0/(N - 1) * (self._sum_sq_col - 2.0 * self._sum_col * self._means + N * self._means**2))
            self._stds[self._stds < eps] = eps
            pickle.dump(obj = {'means': self._means, 'stds': self._stds}, file=save_file, protocol=2)

    # load the means and std devs from pickle file for normalisation of any dataset's samples
    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == '2':
                dict = pickle.load(load_file)
            else:
                dict = pickle.load(load_file, encoding='latin1')
            self._means = dict['means']
            self._stds = dict['stds']

    # normalisation of cols of a sample's 2D array data
    def transform(self, data, fields=None):
        if fields is None:
            fields = range(data.shape[1])   # 0 to no. of cols of data, i.e., ehr_n_var
        ret = 1.0 * data
        for col in fields:  # iterate over the cols of data and normalise each using the mean and std dev of the col
            ret[:, col] = (data[:, col] - self._means[col]) / self._stds[col]
        return ret

class Normalizer2:
    def __init__(self):
        self._means = None
        self._stds = None
        
        self._sum_row = None
        self._sum_sq_row = None
        self._count = 0

    # calculate count, sum and sum of squares of the PPG signals' values over all samples in training set
    def _feed_data(self, data):
        data = np.array(data)
        self._count += data.shape[0]                     # running count of number of PPG signals; scalar
        if self._sum_row is None:
            self._sum_row = np.sum(data, axis=0)
            self._sum_sq_row = np.sum(data**2, axis=0)
        else:
            self._sum_row += np.sum(data, axis=0)        # running sum of the values at each timestamp; 1D array of length 1250
            self._sum_sq_row += np.sum(data**2, axis=0)  # running sum of squares of the values at each timestamp; 1D array of length 1250

    # calculate the means and std devs considering all samples in training set and save them to a pickle file
    def _save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_row
            self._stds = np.sqrt(1.0/(N - 1) * (self._sum_sq_row - 2.0 * self._sum_row * self._means + N * self._means**2))
            self._stds[self._stds < eps] = eps
            pickle.dump(obj = {'means': self._means, 'stds': self._stds}, file=save_file, protocol=2)

    # load the means and std devs from pickle file for normalisation of any dataset's samples
    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == '2':
                dict = pickle.load(load_file)
            else:
                dict = pickle.load(load_file, encoding='latin1')
            self._means = dict['means']
            self._stds = dict['stds']

    # normalisation the values in a sample's PPG signal (2D array of shape (1, 1250))
    def transform(self, data, fields=None):
        if fields is None:
            fields = range(data.shape[1])   # 0 to 1249
        ret = 1.0 * data
        for timestamp in fields:  # iterate over the rows of data and normalise the elements using the mean and std dev
            ret[:, timestamp] = (data[:, timestamp] - self._means[timestamp]) / self._stds[timestamp]
        return ret  # same shape as data