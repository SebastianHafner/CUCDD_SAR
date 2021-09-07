import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from abc import ABC, abstractmethod
from utils import sentinel1_helpers, dataset_helpers, geofiles, label_helpers, config
import scipy
from tqdm import tqdm


class ChangeDetectionMethod(ABC):

    def __init__(self, name: str):
        self.name = name

    @ abstractmethod
    # returns binary array of changes
    def change_detection(self, aoi_id: str) -> np.ndarray:
        pass


class ChangeDatingMethod(ChangeDetectionMethod):

    def __init__(self, name: str):
        super().__init__(name)

    @ abstractmethod
    # returns int array where numbers correspond to change date (index in dates list)
    def change_dating(self, aoi_id: str) -> np.ndarray:
        pass

    @staticmethod
    def _mse(y: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.square(y_hat - y), axis=-1) / y.shape[-1]


class StepFunctionModel(ChangeDatingMethod):

    def __init__(self, band: str, error_multiplier: int = 3, min_diff: float = 0.2, min_segment_length: int = 2,
                 noise_reduction: bool = True):
        super().__init__('stepfunction')
        self.fitted_aoi = None
        # index when changed occurred in the time series
        # (no change is index 0 and length_ts for non-urban and urban, respectively)
        self.cached_fit = None
        self.length_ts = None
        self.band = band
        self.error_multiplier = error_multiplier
        self.min_diff = min_diff
        self.min_segment_length = min_segment_length
        self.noise_reduction = noise_reduction

    def _fit(self, aoi_id: str):
        if self.fitted_aoi == aoi_id:
            return

        timeseries = dataset_helpers.get_timeseries(aoi_id, config.include_masked())
        self.length_ts = len(timeseries)

        data_cube = sentinel1_helpers.load_sentinel1_band_timeseries(aoi_id, self.band, config.include_masked())

        errors = []
        mean_diffs = []

        # compute mse for stable fit
        mean = np.mean(data_cube, axis=-1)
        pred_stable = np.repeat(mean[:, :, np.newaxis], len(timeseries), axis=-1)
        error_stable = self._mse(data_cube, pred_stable)

        # break point detection
        for i in range(self.min_segment_length, len(timeseries) - self.min_segment_length):

            # compute predicted
            presegment = data_cube[:, :, :i]
            mean_presegment = np.mean(presegment, axis=-1)
            pred_presegment = np.repeat(mean_presegment[:, :, np.newaxis], i, axis=-1)

            postsegment = data_cube[:, :, i:]
            mean_postsegment = np.mean(postsegment, axis=-1)
            pred_postsegment = np.repeat(mean_postsegment[:, :, np.newaxis], len(timeseries) - i, axis=-1)

            # maybe use absolute value here
            mean_diffs.append(mean_postsegment - mean_presegment)

            pred_probs_break = np.concatenate((pred_presegment, pred_postsegment), axis=-1)
            mse_break = self._mse(data_cube, pred_probs_break)
            errors.append(mse_break)

        errors = np.stack(errors, axis=-1)
        best_fit = np.argmin(errors, axis=-1)

        min_error_break = np.min(errors, axis=-1)
        change_candidate = min_error_break * self.error_multiplier < error_stable

        mean_diffs = np.stack(mean_diffs, axis=-1)
        m, n = mean_diffs.shape[:2]
        mean_diff = mean_diffs[np.arange(m)[:, None], np.arange(n), best_fit]
        change = np.logical_and(change_candidate, mean_diff > self.min_diff)

        if self.noise_reduction:
            kernel = np.ones((3, 3), dtype=np.uint8)
            change_count = scipy.signal.convolve2d(change, kernel, mode='same', boundary='fill', fillvalue=0)
            noise = change_count == 1
            change[noise] = 0

        self.cached_fit = np.where(change, best_fit + self.min_segment_length, 0)
        self.fitted_aoi = aoi_id

    def change_detection(self, aoi_id: str) -> np.ndarray:
        self._fit(aoi_id)

        # convert to change date product to change detection (0 and length_ts is no change)
        change = self.cached_fit != 0

        return np.array(change).astype(np.bool)

    def change_dating(self, aoi_id: str, config_name: str = None) -> np.ndarray:
        self._fit(aoi_id)

        return np.array(self.cached_fit).astype(np.uint8)

    @ staticmethod
    def exponential_distribution(x: np.ndarray, la: float = 0.25) -> np.ndarray:
        return la * np.e ** (-la * x)


if __name__ == '__main__':
    model = StepFunctionModel()
    change = model.change_detection('spacenet7', 'L15-0566E-1185N_2265_3451_13')
