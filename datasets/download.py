"""
Datasets module
"""
import logging
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from fire import Fire
import patoolib
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
from http_utils import download, url_file_name

DATASETS_PATH = os.path.dirname( __file__ )

def download_tourism_dataset():
    """
    Download Tourism dataset.
    """
    DATASET_URL = 'https://robjhyndman.com/data/27-3-Athanasopoulos1.zip'

    DATASET_PATH = os.path.join(DATASETS_PATH, 'tourism')
    DATASET_FILE_PATH = os.path.join(DATASET_PATH, url_file_name(DATASET_URL))

    if os.path.isdir(DATASET_PATH):
        logging.info(f'skip: {DATASET_PATH} directory already exists.')
        return

    download(DATASET_URL, DATASET_FILE_PATH)
    patoolib.extract_archive(DATASET_FILE_PATH, outdir=DATASET_PATH)


def download_traffic_dataset():
    """
    Download Traffic dataset.
    """
    DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip'

    DATASET_PATH = os.path.join(DATASETS_PATH, 'traffic')
    DATASET_FILE_PATH = os.path.join(DATASET_PATH, url_file_name(DATASET_URL))

    CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'traffic.npz')
    DATES_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'dates.npz')

    if os.path.isdir(DATASET_PATH):
        logging.info(f'skip: {DATASET_PATH} directory already exists.')
        return

    download(DATASET_URL, DATASET_FILE_PATH)
    patoolib.extract_archive(DATASET_FILE_PATH, outdir=DATASET_PATH)
    with open(os.path.join(DATASET_PATH, 'PEMS_train'), 'r') as f:
        train_raw_data = f.readlines()
    with open(os.path.join(DATASET_PATH, 'PEMS_test'), 'r') as f:
        test_raw_data = f.readlines()
    with open(os.path.join(DATASET_PATH, 'randperm'), 'r') as f:
        permutations = f.readlines()
    permutations = np.array(permutations[0].rstrip()[1:-1].split(' ')).astype(np.int32)

    raw_data = train_raw_data + test_raw_data

    # The assumption below does NOT affect the results, because the splits we use in the publication are
    # based on either dates within the first 6 months, where the labels are aligned or on
    # the last values of dataset. Thus, there should not be any confusion with misaligned split points.
    #
    # Dataset dates issue:
    #
    # From the dataset description [https://archive.ics.uci.edu/ml/datasets/PEMS-SF] :
    # "The measurements cover the period from Jan. 1st 2008 to Mar. 30th 2009"
    # and
    # "We remove public holidays from the dataset, as well
    # as two days with anomalies (March 8th 2009 and March 9th 2008)".
    #
    # Based on provided labels, which are days of week, the sequence of days had only 10 gaps by 1 day,
    # where the first 6 correspond to a holiday or anomalous day, but the other 4 gaps happen on "random" dates,
    # meaning we could not find any holiday or the mentioned anomalous days around those dates.
    #
    # More over, the number of days between 2008-01-01 and 2009-03-30 is 455, with only 10 gaps it's
    # not possible to fill dates up to 2009-03-30, it should be 15 gaps (if 2009-01-01 is included, 14 otherwise).
    #
    # Thus, it is not clear if either labels are not correct or the dataset description.
    #
    # Since we are not using any covariates and the split dates after the first 6 months we just fill the gaps with
    # the most common holidays, it does not have any impact on the split points anyway.
    current_date = datetime.strptime('2008-01-01', '%Y-%m-%d')
    excluded_dates = [
        datetime.strptime('2008-01-01', '%Y-%m-%d'),
        datetime.strptime('2008-01-21', '%Y-%m-%d'),
        datetime.strptime('2008-02-18', '%Y-%m-%d'),
        datetime.strptime('2008-03-09', '%Y-%m-%d'),
        datetime.strptime('2008-05-26', '%Y-%m-%d'),
        datetime.strptime('2008-07-04', '%Y-%m-%d'),
        datetime.strptime('2008-09-01', '%Y-%m-%d'),
        datetime.strptime('2008-10-13', '%Y-%m-%d'),
        datetime.strptime('2008-11-11', '%Y-%m-%d'),
        datetime.strptime('2008-11-27', '%Y-%m-%d'),
        datetime.strptime('2008-12-25', '%Y-%m-%d'),
        datetime.strptime('2009-01-01', '%Y-%m-%d'),
        datetime.strptime('2009-01-19', '%Y-%m-%d'),
        datetime.strptime('2009-02-16', '%Y-%m-%d'),
        datetime.strptime('2009-03-08', '%Y-%m-%d'),
    ]
    dates = []
    np_array = []
    for i in tqdm(range(len(permutations))):
        # values
        matrix = raw_data[np.where(permutations == i + 1)[0][0]].rstrip()[1:-1]
        daily = []
        for row_vector in matrix.split(';'):
            daily.append(np.array(row_vector.split(' ')).astype(np.float32))
        daily = np.array(daily)
        if len(np_array) == 0:
            np_array = daily
        else:
            np_array = np.concatenate([np_array, daily], axis=1)

        # dates
        while current_date in excluded_dates:  # skip those in excluded dates
            current_date = current_date + timedelta(days=1)
        dates.extend([(current_date + timedelta(hours=i + 1)).strftime('%Y-%m-%d %H') for i in range(24)])
        current_date = current_date + timedelta(days=1)

    # aggregate 10 minutes events to hourly
    hourly = np.array([list(map(np.mean, zip(*(iter(lane),) * 6))) for lane in tqdm(np_array)])
    logging.info(f'Caching data {hourly.shape} to {CACHE_FILE_PATH}')
    hourly.dump(CACHE_FILE_PATH)
    logging.info(f'Caching dates to {DATES_CACHE_FILE_PATH}')
    np.array(dates).dump(DATES_CACHE_FILE_PATH)

def download_electricity_dataset():
    """
    Download Electricity dataset.
    """
    DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'

    DATASET_DIR = os.path.join(DATASETS_PATH, 'electricity')
    DATASET_FILE_PATH = os.path.join(DATASET_DIR, url_file_name(DATASET_URL))
    RAW_DATA_FILE_PATH = os.path.join(DATASET_DIR, 'LD2011_2014.txt')

    CACHE_FILE_PATH = os.path.join(DATASET_DIR, 'electricity.npz')
    DATES_CACHE_FILE_PATH = os.path.join(DATASET_DIR, 'dates.npz')

    if os.path.isdir(DATASET_DIR):
        logging.info(f'skip: {DATASET_DIR} directory already exists.')
        return
    download(DATASET_URL, DATASET_FILE_PATH)
    patoolib.extract_archive(DATASET_FILE_PATH, outdir=DATASET_DIR)
    with open(RAW_DATA_FILE_PATH, 'r') as f:
        raw = f.readlines()

    # based on data downloaded by script:
    # https://github.com/rofuyu/exp-trmf-nips16/blob/master/python/exp-scripts/datasets/download-data.sh
    # the first year of data was ignored.
    # The raw data frequency is 15 minutes, thus we ignore header, first record, and 4 * 24 * 365 data points
    header = 1
    ignored_first_values = header + (365 * 24 * 4)
    parsed_values = list(map(lambda raw_line: raw_line.replace(',', '.').strip().split(';')[1:],
                             raw[ignored_first_values:]))
    data = np.array(parsed_values).astype(np.float32)

    # aggregate to hourly
    aggregated = []
    for i in tqdm(range(0, data.shape[0], 4)):
        aggregated.append(data[i:i + 4, :].sum(axis=0))
    aggregated = np.array(aggregated)

    dataset = aggregated.T  # use time step as second dimension.
    logging.info(f'Caching matrix {dataset.shape} to {CACHE_FILE_PATH}')
    dataset.dump(CACHE_FILE_PATH)
    logging.info(f'Caching dates to {DATES_CACHE_FILE_PATH}')
    dates = list(map(lambda raw_line: raw_line.replace(',', '.').strip().split(';')[0], raw[ignored_first_values:]))
    # ignore first hour, for its values are aggregated to the next hour.
    np.unique(list(
        map(lambda s: datetime.strptime(s[1:-1], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H'), dates)))[1:].dump(
        DATES_CACHE_FILE_PATH)

def build():
    """
    Download all datasets.
    """

    logging.info('\n\nTourism Dataset')
    download_tourism_dataset()

    logging.info('\n\nElectricity Dataset')
    download_electricity_dataset()

    logging.info('\n\nTraffic Dataset')
    download_traffic_dataset()

if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire()