import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from abc import ABC, abstractmethod
from utils import input_helpers, dataset_helpers, geofiles, label_helpers, config
import scipy
from tqdm import tqdm


class ChangeDetectionMethod(ABC):

    def __init__(self, name: str, config_name: str = None):
        self.name = name

    @ abstractmethod
    # returns binary array of changes
    def change_detection(self, dataset: str, aoi_id: str) -> np.ndarray:
        pass


class ChangeDatingMethod(ChangeDetectionMethod):

    def __init__(self, name: str):
        super().__init__(name)

    @ abstractmethod
    # returns int array where numbers correspond to change date (index in dates list)
    def change_dating(self, dataset: str, aoi_id: str) -> np.ndarray:
        pass

    @staticmethod
    def _mse(y: np.ndarray, y_hat: np.ndarray) -> float:
        return np.sum(np.square(y_hat - y), axis=-1) / y.shape[-1]


class StepFunctionModel(ChangeDatingMethod):

    def __init__(self, error_multiplier: int = 3, min_prob_diff: float = 0.2, min_segment_length: int = 2,
                 improve_last: bool = False, improve_first: bool = False, noise_reduction: bool = True):
        super().__init__('stepfunction')
        self.fitted_dataset = None
        self.fitted_aoi = None
        # index when changed occurred in the time series
        # (no change is index 0 and length_ts for non-urban and urban, respectively)
        self.cached_fit = None
        self.length_ts = None

        self.error_multiplier = error_multiplier
        self.min_prob_diff = min_prob_diff
        self.min_segment_length = min_segment_length
        self.improve_last = improve_last
        self.improve_first = improve_first
        self.noise_reduction = noise_reduction

    def _fit(self, dataset: str, aoi_id: str):
        if dataset == self.fitted_dataset and self.fitted_aoi == aoi_id:
            return

        timeseries = dataset_helpers.get_timeseries(dataset, aoi_id, config.include_masked())
        self.length_ts = len(timeseries)

        probs_cube = input_helpers.load_input_timeseries(dataset, aoi_id, config.include_masked())

        errors = []
        mean_diffs = []

        # compute mse for stable fit
        mean_prob = np.mean(probs_cube, axis=-1)
        pred_prob_stable = np.repeat(mean_prob[:, :, np.newaxis], len(timeseries), axis=-1)
        error_stable = self._mse(probs_cube, pred_prob_stable)

        if self.improve_first:
            coefficients = self.exponential_distribution(np.arange(self.length_ts))
            probs_cube = probs_cube.transpose((2, 0, 1))
            probs_cube = coefficients[:, np.newaxis, np.newaxis] * probs_cube
            probs_exp = np.sum(probs_cube, axis=0)
            probs_cube[0, :, :] = probs_exp
            probs_cube = probs_cube.transpose((1, 2, 0))

        if self.improve_last:
            coefficients = self.exponential_distribution(np.arange(self.length_ts))[::-1]
            probs_cube_exp = coefficients[:, np.newaxis, np.newaxis] * probs_cube.transpose((2, 0, 1))
            probs_exp = np.sum(probs_cube_exp, axis=0)
            probs_cube[:, :, -1] = probs_exp


        # break point detection
        for i in range(self.min_segment_length, len(timeseries) - self.min_segment_length):

            # compute predicted
            probs_presegment = probs_cube[:, :, :i]
            mean_prob_presegment = np.mean(probs_presegment, axis=-1)
            pred_probs_presegment = np.repeat(mean_prob_presegment[:, :, np.newaxis], i, axis=-1)

            probs_postsegment = probs_cube[:, :, i:]
            mean_prob_postsegment = np.mean(probs_postsegment, axis=-1)
            pred_probs_postsegment = np.repeat(mean_prob_postsegment[:, :, np.newaxis], len(timeseries) - i, axis=-1)

            # maybe use absolute value here
            mean_diffs.append(mean_prob_postsegment - mean_prob_presegment)

            pred_probs_break = np.concatenate((pred_probs_presegment, pred_probs_postsegment), axis=-1)
            mse_break = self._mse(probs_cube, pred_probs_break)
            errors.append(mse_break)

        errors = np.stack(errors, axis=-1)
        best_fit = np.argmin(errors, axis=-1)

        min_error_break = np.min(errors, axis=-1)
        change_candidate = min_error_break * self.error_multiplier < error_stable

        mean_diffs = np.stack(mean_diffs, axis=-1)
        m, n = mean_diffs.shape[:2]
        mean_diff = mean_diffs[np.arange(m)[:, None], np.arange(n), best_fit]
        change = np.logical_and(change_candidate, mean_diff > self.min_prob_diff)

        if self.noise_reduction:
            kernel = np.ones((3, 3), dtype=np.uint8)
            change_count = scipy.signal.convolve2d(change, kernel, mode='same', boundary='fill', fillvalue=0)
            noise = change_count == 1
            change[noise] = 0

        self.cached_fit = np.where(change, best_fit + self.min_segment_length, 0)

        self.fitted_dataset = dataset
        self.fitted_aoi = aoi_id

    def change_detection(self, dataset: str, aoi_id: str) -> np.ndarray:
        self._fit(dataset, aoi_id)

        # convert to change date product to change detection (0 and length_ts is no change)
        change = self.cached_fit != 0

        return np.array(change).astype(np.bool)

    def change_dating(self, dataset: str, aoi_id: str, config_name: str = None) -> np.ndarray:
        self._fit(dataset, aoi_id)

        return np.array(self.cached_fit).astype(np.uint8)

    @ staticmethod
    def exponential_distribution(x: np.ndarray, la: float = 0.25) -> np.ndarray:
        return la * np.e ** (-la * x)


if __name__ == '__main__':
    model = StepFunctionModel()
    change = model.change_detection('spacenet7', 'L15-0566E-1185N_2265_3451_13')
