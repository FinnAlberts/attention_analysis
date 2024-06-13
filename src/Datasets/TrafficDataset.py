"""
Traffic Dataset
"""
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np

"""
Hourly aggregated dataset from https://archive.ics.uci.edu/ml/datasets/PEMS-SF

As it is used in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
Dataset was also compared with the one built by the TRMF paper's author:
https://github.com/rofuyu/exp-trmf-nips16/blob/master/python/exp-scripts/datasets/download-data.sh
"""

# r"../../datasets/traffic"
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets', 'traffic')

CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'traffic.npz')
DATES_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'dates.npz')

@dataclass()
class TrafficMeta:
    horizon = 24
    lanes = 963
    seasonal_pattern = 'Hourly'
    frequency = 24 * 7  # week, same time


@dataclass()
class TrafficDataset:
    ids: np.ndarray
    values: np.ndarray
    dates: np.ndarray

    @staticmethod
    def load() -> 'TrafficDataset':
        """
        Load Traffic dataset from cache.
        :return:
        """
        values = np.load(CACHE_FILE_PATH, allow_pickle=True)
        return TrafficDataset(
            ids=np.array(list(range(len(values)))),
            values=values.T,
            dates=np.load(DATES_CACHE_FILE_PATH, allow_pickle=True))

    def split_by_date(self, cut_date: str, include_cut_date: bool = True) -> Tuple['TrafficDataset', 'TrafficDataset']:
        """
        Split dataset by date.

        :param cut_date: Cut date in "%Y-%m-%d %H" format
        :param include_cut_date: Include cut_date in the split if true, not otherwise.
        :return: Left and right parts of the split.
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
        return TrafficDataset(ids=self.ids,
                              values=self.values[left_indices],
                              dates=self.dates[left_indices]), \
               TrafficDataset(ids=self.ids,
                              values=self.values[right_indices],
                              dates=self.dates[right_indices])

    def split(self, cut_point: int) -> Tuple['TrafficDataset', 'TrafficDataset']:
        """
        Split dataset by cut point.

        :param cut_point: Cut index.
        :return: Two parts of dataset: the left part contains all points before the cut point
        and the right part contains all datpoints on and after the cut point.
        """
        return TrafficDataset(ids=self.ids,
                              values=self.values[:cut_point],
                              dates=self.dates[:cut_point]), \
               TrafficDataset(ids=self.ids,
                              values=self.values[cut_point:],
                              dates=self.dates[cut_point:])

    def time_points(self):
        return self.dates.shape[0]