from pathlib import Path
import yaml


def settings() -> dict:
    with open(str(Path.cwd() / 'settings.yaml')) as file:
        s = yaml.load(file, Loader=yaml.FullLoader)
    return s


SETTINGS = settings()


def dataset_name() -> Path:
    return SETTINGS['DATASET_NAME']


# dataset paths
def root_path() -> Path:
    return Path(SETTINGS['PATHS']['ROOT'])


def output_path() -> Path:
    return Path(SETTINGS['PATHS']['OUTPUT'])


def dataset_path() -> Path:
    return Path(SETTINGS['PATHS']['DATASET'])


# path to origin SpaceNet7 dataset
def spacenet7_path() -> Path:
    return Path(SETTINGS['PATHS']['SPACENET7'])


def subset_activated() -> bool:
    return SETTINGS['SUBSET']['ACTIVATE']


def subset_aois() -> list:
    return SETTINGS['SUBSET']['AOI_IDS']


def min_timeseries_length() -> bool:
    return SETTINGS['MIN_TIMESERIES_LENGTH']


def consistent_timeseries_length() -> bool:
    return SETTINGS['TIMESERIES']['CONSISTENT_LENGTH']
