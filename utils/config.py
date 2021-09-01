from pathlib import Path
import yaml


def settings() -> dict:
    with open(str(Path.cwd() / 'settings.yaml')) as file:
        s = yaml.load(file, Loader=yaml.FullLoader)
    return s


SETTINGS = settings()


# dataset names
def spacenet7_dataset_name() -> str:
    return Path(SETTINGS['DATASET_NAMES']['SPACENET7'])


def oscd_dataset_name() -> str:
    return Path(SETTINGS['DATASET_NAMES']['OSCD'])


# dataset paths
def root_path() -> Path:
    return Path(SETTINGS['PATHS']['ROOT_PATH'])


# path to origin SpaceNet7 dataset
def spacenet7_path() -> Path:
    return Path(SETTINGS['PATHS']['SPACENET7_PATH'])


def oscd_path() -> Path:
    return Path(SETTINGS['PATHS']['OSCD_PATH'])


def include_masked() -> bool:
    return SETTINGS['INCLUDE_MASKED_DATA']


# sensor settings
def subset_activated(dataset: str) -> bool:
    if dataset == 'spacenet7':
        return SETTINGS['SUBSET_SPACENET7']['ACTIVATE']
    else:
        return SETTINGS['SUBSET_OSCD']['ACTIVATE']


def subset_aois(dataset: str) -> list:
    if dataset == 'spacenet7':
        return SETTINGS['SUBSET_SPACENET7']['AOI_IDS']
    else:
        return SETTINGS['SUBSET_OSCD']['AOI_IDS']
