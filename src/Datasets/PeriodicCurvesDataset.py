
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Union

import numpy as np

@dataclass()
class PeriodicCurvesMeta:
    horizon = 100

@dataclass()
class PeriodicCurvesDataset:
    ids: np.ndarray
    values: np.ndarray
    dates: np.ndarray

    @staticmethod
    def load(len_timeseries: int,
             A_array: Union[np.ndarray, list]) -> 'PeriodicCurvesDataset':
        """
        Loads periodic curves from a time series.
        :param len_timeseries: total length of time series (must be greater than 120)
        :param A_array: numpy array of shape (4, n_timeseries, 1) or list of length 4 with numpy arrays of shape (n_timeseries, 1) as elements
        """

        A1, A2, A3, A4 = A_array
        X = np.arange(0, len_timeseries, dtype=np.float32)

        c1 = A1 * np.sin(np.pi * X[None, :12] / 6) + 72
        c2 = A2 * np.sin(np.pi * X[None, 12:24] / 6) + 72
        c3 = A3 * np.sin(np.pi * X[None, 24:96] / 6) + 72
        c4 = A4 * np.sin(np.pi * X[None, 96:120] / 12) + 72

        fX = np.concatenate([c1, c2, c3, c4], axis=-1) # concat components
        fX = np.tile(fX, (len_timeseries + 120) // 120) # periodic repeats
        fX = fX[:, :len_timeseries]  # constrain total length
        fX = fX + np.random.randn(*fX.shape)  # add noise

        return PeriodicCurvesDataset(
            ids=np.arange(0, A1.shape[0]),
            values=fX,
            dates=X
        )

    @staticmethod
    def load_multivariate(len_timeseries: int, n_timeseries: int) -> 'PeriodicCurvesDataset':
        A1, A2, A3 = np.random.rand(3, n_timeseries, 1) * 60
        A4 = np.maximum(A1, A2)
        return PeriodicCurvesDataset.load(len_timeseries, [A1, A2, A3, A4])

    @staticmethod
    def load_univariate(len_timeseries: int, A1=5, A2=100, A3=3) -> 'PeriodicCurvesDataset':
        A1, A2, A3 = np.array([[[A1]], [[A2]], [[A3]]])
        A4 = np.maximum(A1, A2)
        return PeriodicCurvesDataset.load(len_timeseries, [A1, A2, A3, A4])

    def split(self, cut_point: int) -> Tuple['PeriodicCurvesDataset', 'PeriodicCurvesDataset']:
        """
        Split dataset by cut point.

        :param cut_point: Cut index.
        :return: Two parts of dataset: the left part contains all points before the cut point
        and the right part contains all datpoints on and after the cut point.
        """
        return PeriodicCurvesDataset(ids=self.ids,
                              values=self.values[:, :cut_point],
                              dates=self.dates[:cut_point]), \
               PeriodicCurvesDataset(ids=self.ids,
                              values=self.values[:, cut_point:],
                              dates=self.dates[cut_point:])

    def time_points(self):
        return self.dates.shape[0]