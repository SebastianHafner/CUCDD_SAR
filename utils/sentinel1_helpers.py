from pathlib import Path
from utils import geofiles, dataset_helpers, config
import numpy as np


def convert_range(img: np.ndarray, old_min: float, old_max: float, new_min: float, new_max: float) -> np.ndarray:
    return (((img - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


def load_sentinel1(aoi_id: str, year: int, month: int) -> np.ndarray:
    file = dataset_helpers.dataset_path() / aoi_id / 'sentinel1' / f'sentinel1_{aoi_id}_{year}_{month:02d}.tif'
    img, _, _ = geofiles.read_tif(file)
    img = convert_range(img, 0, 1, -25, 0)
    return img


def load_sentinel1_band(aoi_id: str, year: int, month: int, band: str) -> np.ndarray:
    img = load_sentinel1(aoi_id, year, month)
    bands = ['VV', 'VH']
    band_index = bands.index(band)
    img = img[:, :, band_index]
    return img


def load_sentinel1_timeseries(aoi_id: str, include_masked_data: bool = False, ts_extension: int = 0) -> np.ndarray:
    dates = dataset_helpers.get_timeseries(aoi_id, include_masked_data)

    yx_shape = dataset_helpers.get_yx_size(aoi_id)
    n = len(dates)
    s1_ts = np.zeros((*yx_shape, get_n_bands(), n + 2 * ts_extension), dtype=np.float32)

    # fill in time series value
    for i, (year, month, *_) in enumerate(dates):
        s1 = load_sentinel1(aoi_id, year, month)
        s1_ts[:, :, :, i + ts_extension] = s1

    # padd start and end
    if ts_extension != 0:
        start_pred = s1_ts[:, :, ts_extension]
        start_extension = np.repeat(start_pred[:, :, np.newaxis], ts_extension, axis=2)
        s1_ts[:, :, :ts_extension] = start_extension

        end_index = ts_extension + n - 1
        end_pred = s1_ts[:, :, end_index]
        end_extension = np.repeat(end_pred[:, :, np.newaxis], ts_extension, axis=2)
        s1_ts[:, :, end_index + 1:] = end_extension

    return s1_ts


def load_sentinel1_band_timeseries(aoi_id: str, band: str, include_masked_data: bool = False,
                                   ts_extension: int = 0) -> np.ndarray:
    s1_ts = load_sentinel1_timeseries(aoi_id, include_masked_data, ts_extension)
    band_index = get_band_index(band)

    return s1_ts[:, :, band_index, ]


def get_n_bands() -> int:
    md = dataset_helpers.metadata()
    return len(md['bands'])


def get_band_index(band: str) -> int:
    md = dataset_helpers.metadata()
    return md['bands'].index(band)


if __name__ == '__main__':
    img = np.array([0, 0.2, 0.5, 1])
    img_new = convert_range(img, 0, 1, -25, 0)
    print(img_new)
