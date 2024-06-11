"""
Electricity Dataset
"""
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np

"""
Hourly aggregated dataset from https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

As it is used in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
Dataset was also compared with the one built by the TRMF paper's author:
https://github.com/rofuyu/exp-trmf-nips16/blob/master/python/exp-scripts/datasets/download-data.sh
"""

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets', 'electricity')

CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'electricity.npz')
DATES_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'dates.npz')

@dataclass()
class ElectricityMeta:
    horizon = 24
    clients = 370
    time_steps = 26304
    seasonal_pattern = 'Hourly'
    frequency = 24

@dataclass()
class ElectricityDataset:
    ids: np.ndarray
    values: np.ndarray
    dates: np.ndarray

    @staticmethod
    def load() -> 'ElectricityDataset':
        """
        Load Electricity dataset from cache.
        """
        value = np.load(CACHE_FILE_PATH, allow_pickle=True)
        return ElectricityDataset(
            ids=np.array(list(range(len(value)))),
            values=np.load(CACHE_FILE_PATH, allow_pickle=True),
            dates=np.load(DATES_CACHE_FILE_PATH, allow_pickle=True))

    def split_by_date(self, cut_date: str, include_cut_date: bool = True) -> Tuple['ElectricityDataset', 'ElectricityDataset']:
        """
        Split dataset by date.

        :param cut_date: Cut date in "%Y-%m-%d %H" format
        :param include_cut_date: Include cut_date in the split if true, not otherwise.
        :return: Two parts of dataset: the left part contains all points before the cut point
        and the right part contains all datpoints on and after the cut point.
        """
        date = datetime.strptime(cut_date, '%Y-%m-%d %H')
        left_indices = []
        right_indices = []
        for i, p in enumerate(self.dates):
            record_date = datetime.strptime(p, '%Y-%m-%d %H')
            if record_date < date or (include_cut_date and record_date == date):
                left_indices.append(i)
            else:
                right_indices.append(i)
        return ElectricityDataset(ids=self.ids,
                                  values=self.values[:, left_indices],
                                  dates=self.dates[left_indices]), \
               ElectricityDataset(ids=self.ids,
                                  values=self.values[:, right_indices],
                                  dates=self.dates[right_indices])

    def split(self, cut_point: int) -> Tuple['ElectricityDataset', 'ElectricityDataset']:
        """
        Split dataset by cut point.

        :param cut_point: Cut index.
        :return: Two parts of dataset: left contains all points before the cut point
        and the right part contains all datpoints on and after the cut point.
        """
        return ElectricityDataset(ids=self.ids,
                                  values=self.values[:, :cut_point],
                                  dates=self.dates[:cut_point]), \
               ElectricityDataset(ids=self.ids,
                                  values=self.values[:, cut_point:],
                                  dates=self.dates[cut_point:])

    def time_points(self):
        return self.dates.shape[0]