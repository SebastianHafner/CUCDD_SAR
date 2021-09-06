from pathlib import Path
from utils import geofiles, dataset_helpers, config
import numpy as np


def load_sentinel1(dataset: str, aoi_id: str, year: int, month: int) -> np.ndarray:
    file = dataset_helpers.dataset_path(dataset) / aoi_id / 'sentinel1' / f'sentinel1_{aoi_id}_{year}_{month:02d}.tif'
    img, _, _ = geofiles.read_tif(file)
    return img


def load_sentinel1_band(dataset: str, aoi_id: str, year: int, month: int, band: str) -> np.ndarray:
    img = load_sentinel1(dataset, aoi_id, year, month)
    bands = ['VV', 'VH']
    band_index = bands.index(band)
    img = img[:, :, band_index]
    return img


def load_input_timeseries(dataset: str, aoi_id: str, include_masked_data: bool = False,
                          ts_extension: int = 0) -> np.ndarray:
    dates = dataset_helpers.get_timeseries(dataset, aoi_id, include_masked_data)

    yx_shape = dataset_helpers.get_yx_size(dataset, aoi_id)
    n = len(dates)
    pred_ts = np.zeros((dataset_helpers.get_n_bands(dataset), *yx_shape, n + 2 * ts_extension), dtype=np.float32)

    # fill in time series value
    for i, (year, month, *_) in enumerate(dates):
        pred = load_sentinel1(dataset, aoi_id, year, month)
        pred_ts[:, :, i + ts_extension] = pred

    # padd start and end
    if ts_extension != 0:
        start_pred = pred_ts[:, :, ts_extension]
        start_extension = np.repeat(start_pred[:, :, np.newaxis], ts_extension, axis=2)
        pred_ts[:, :, :ts_extension] = start_extension

        end_index = ts_extension + n - 1
        end_pred = pred_ts[:, :, end_index]
        end_extension = np.repeat(end_pred[:, :, np.newaxis], ts_extension, axis=2)
        pred_ts[:, :, end_index + 1:] = end_extension

    return pred_ts


def load_input_in_timeseries(dataset: str, aoi_id: str, index: int, include_masked_data: bool,
                             ignore_bad_data: bool = True) -> np.ndarray:
    dates = dataset_helpers.get_timeseries(dataset, aoi_id, include_masked_data, ignore_bad_data)
    year, month, *_ = dates[index]
    pred = load_prediction(dataset, aoi_id, year, month)
    return pred


def load_features_in_timeseries(dataset: str, aoi_id: str, index: int, include_masked_data: bool) -> np.ndarray:
    dates = dataset_helpers.get_timeseries(dataset, aoi_id, include_masked_data)
    year, month, *_ = dates[index]
    features = load_features(dataset, aoi_id, year, month)
    return features


def load_features(dataset: str, aoi_id: str, year: int, month: int) -> np.ndarray:
    predictions_path = dataset_helpers.dataset_path(dataset) / aoi_id / dataset_helpers.config_name()
    file = predictions_path / f'features_{aoi_id}_{year}_{month:02d}.npy'
    features = np.load(str(file))
    return features


def load_satellite_timeseries(dataset: str, aoi_id: str, satellite: str, band: str):
    dates = dataset_helpers.get_timeseries(dataset, aoi_id, dataset_helpers.include_masked())

    yx_shape = dataset_helpers.get_yx_size(dataset, aoi_id)
    data = np.zeros((*yx_shape, len(dates)), dtype=np.float32)

    # fill in time series value
    for i, (year, month, *_) in enumerate(dates):
        if satellite == 'sentinel1':
            img = load_sentinel1(dataset, aoi_id, year, month, band)
        else:
            img = load_sentinel2(dataset, aoi_id, year, month, band)
        data[:, :, i] = img

    return data


if __name__ == '__main__':
    # predictions = load_prediction_timeseries('spacenet7', 'L15-0331E-1257N_1327_3160_13')
    a = np.array([False, np.NaN])
    a = np.ma.array(a, mask=np.isnan(a))
    print(np.any(a))
