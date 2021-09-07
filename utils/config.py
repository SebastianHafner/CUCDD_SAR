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


# dataset paths
def root_path() -> Path:
    return Path(SETTINGS['PATHS']['ROOT_PATH'])


# path to origin SpaceNet7 dataset
def spacenet7_path() -> Path:
    return Path(SETTINGS['PATHS']['SPACENET7_PATH'])


def include_masked() -> bool:
    return SETTINGS['INCLUDE_MASKED_DATA']


# sensor settings
def subset_activated() -> bool:
    return SETTINGS['SUBSET']['ACTIVATE']


def subset_aois() -> list:
    return SETTINGS['SUBSET']['AOI_IDS']
