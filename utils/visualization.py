import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import geofiles, dataset_helpers, label_helpers, mask_helpers, sentinel1_helpers, config
import numpy as np
from pathlib import Path
from matplotlib import cm


class DateColorMap(object):

    def __init__(self, ts_length: int, color_map: str = 'jet'):
        self.ts_length = ts_length
        default_cmap = cm.get_cmap(color_map, ts_length - 1)
        dates_colors = default_cmap(np.linspace(0, 1, ts_length - 1))
        no_change_color = np.array([0, 0, 0, 1])
        cmap_colors = np.zeros((ts_length, 4))
        cmap_colors[0, :] = no_change_color
        cmap_colors[1:, :] = dates_colors
        self.cmap = mpl.colors.ListedColormap(cmap_colors)

    def get_cmap(self):
        return self.cmap

    def get_vmin(self):
        return 0

    def get_vmax(self):
        return self.ts_length


class ChangeConfidenceColorMap(object):

    def __init__(self, color_map: str = 'RdYlGn'):
        n = int(1e3)
        default_cmap = cm.get_cmap(color_map, n - 1)
        confidence_colors = default_cmap(np.linspace(0, 1, n - 1))
        no_change_color = np.array([0, 0, 0, 1])
        cmap_colors = np.zeros((n, 4))
        cmap_colors[0, :] = no_change_color
        cmap_colors[1:, :] = confidence_colors
        self.cmap = mpl.colors.ListedColormap(cmap_colors)

    def get_cmap(self):
        return self.cmap


def plot_optical(ax, aoi_id: str, year: int, month: int, vis: str = 'true_color', rescale_factor: float = 0.4):
    file = config.dataset_path() / aoi_id / 'sentinel2' / f'sentinel2_{aoi_id}_{year}_{month:02d}.tif'
    if not file.exists():
        return
    img, _, _ = geofiles.read_tif(file)
    band_indices = [2, 1, 0] if vis == 'true_color' else [6, 2, 1]
    bands = img[:, :, band_indices] / rescale_factor
    bands = bands.clip(0, 1)
    ax.imshow(bands)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_sar(ax, aoi_id: str, year: int, month: int, vis: str = 'VV', i_range: tuple = None, j_range: tuple = None):
    file = config.dataset_path() / aoi_id / 'sentinel1' / f'sentinel1_{aoi_id}_{year}_{month:02d}.tif'
    if not file.exists():
        return
    img, _, _ = geofiles.read_tif(file)
    band_index = 0 if vis == 'VV' else 1
    bands = img[:, :, band_index]
    bands = bands.clip(0, 1)
    if i_range is not None and j_range is not None:
        i_start, i_end = i_range
        j_start, j_end = j_range
        bands = bands[i_start:i_end, j_start:j_end]
    ax.imshow(bands, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_buildings(ax, aoi_id: str, year: int, month: int):
    buildings = label_helpers.load_label(aoi_id, year, month)
    isnan = np.isnan(buildings)
    buildings = buildings.astype(np.uint8)
    buildings = np.where(~isnan, buildings, 3)
    colors = [(0, 0, 0), (1, 1, 1), (1, 0, 0)]
    cmap = mpl.colors.ListedColormap(colors)
    ax.imshow(buildings, cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_change_label(ax, aoi_id: str):
    change = label_helpers.generate_change_label(aoi_id)
    ax.imshow(change, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_change_date_label(ax, aoi_id: str):
    ts = dataset_helpers.get_timeseries(aoi_id)
    change_date_label = label_helpers.generate_change_date_label(aoi_id)
    cmap = DateColorMap(len(ts))
    ax.imshow(change_date_label, cmap=cmap.get_cmap(), vmin=cmap.get_vmin(), vmax=cmap.get_vmax())
    ax.set_xticks([])
    ax.set_yticks([])


def plot_change_date(ax, arr: np.ndarray, ts_length: int):
    cmap = DateColorMap(ts_length)
    ax.imshow(arr, cmap=cmap.get_cmap(), vmin=cmap.get_vmin(), vmax=cmap.get_vmax())
    ax.set_xticks([])
    ax.set_yticks([])


def plot_change_data_bar(ax, dates: list):
    cb_ticks = [0.5, 1.5] + list(np.arange(len(dates)) + 2.5)
    cmap = DateColorMap(ts_length=len(dates))
    norm = mpl.colors.Normalize(vmin=cmap.get_vmin(), vmax=cmap.get_vmax())
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap.get_cmap(), norm=norm, orientation='horizontal',
                                   ticks=cb_ticks)
    cb.set_label('Change Date (yy-mm)', fontsize=config.fontsize())
    cb_ticklabels = ['NC'] + [dataset_helpers.date2str(d) for d in dates] + ['BUA']
    cb.ax.set_xticklabels(cb_ticklabels, fontsize=config.fontsize())


def plot_blackwhite(ax, img: np.ndarray, cmap: str = 'gray'):
    ax.imshow(img.clip(0, 1), cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_classification(ax, pred: np.ndarray, dataset: str, aoi_id: str):
    label = label_helpers.generate_change_label(dataset, aoi_id).astype(np.bool)
    pred = pred.squeeze().astype(np.bool)
    tp = np.logical_and(pred, label)
    fp = np.logical_and(pred, ~label)
    fn = np.logical_and(~pred, label)

    img = np.zeros(pred.shape, dtype=np.uint8)

    img[tp] = 1
    img[fp] = 2
    img[fn] = 3

    colors = [(0, 0, 0), (1, 1, 1), (142/255, 1, 0), (140/255, 25/255, 140/255)]
    cmap = mpl.colors.ListedColormap(colors)
    ax.imshow(img, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_change_confidence(ax, change: np.ndarray, confidence: np.ndarray, cmap: str = 'RdYlGn'):
    confidence = (confidence + 1e-6) * change
    cmap = ChangeConfidenceColorMap().get_cmap()
    ax.imshow(confidence, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_prediction(ax, dataset: str, aoi_id: str, year: int, month: int):
    if not input_helpers.prediction_is_available(dataset, aoi_id, year, month):
        return
    pred = input_helpers.load_prediction(dataset, aoi_id, year, month)
    ax.imshow(pred.clip(0, 1), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_mask(ax, dataset: str, aoi_id: str, year: int, month: int):
    mask = mask_helpers.load_mask(dataset, aoi_id, year, month)
    ax.imshow(mask.astype(np.uint8), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_planet_monthly_mosaic(ax, aoi_id: str, year: int, month: int, i_range: tuple = None, j_range: tuple = None,
                               marker: tuple = None):
    f = config.spacenet7_path() / 'train' / aoi_id / 'images' / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
    img, *_ = geofiles.read_tif(f)
    if marker is not None:
        i, j = marker
        marker_size = 2
        yes = np.full((marker_size * 2, marker_size * 2, 4), 255)
        yes[:, :, 1:-1] = 0
        img[i - marker_size: i + marker_size, j - marker_size: j + marker_size, :] = yes
    if i_range is not None and j_range is not None:
        i_start, i_end = i_range
        j_start, j_end = j_range
        img = img[i_start:i_end, j_start:j_end, :]
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == '__main__':
    arr = np.array([[0, 0.01, 0.1, 0.89, 0.9, 1, 1, 1]]).flatten()
    # hist, bin_edges = np.histogram(arr, bins=10, range=(0, 1))
    cmap = mpl.cm.get_cmap('Reds')
    norm = mpl.colors.Normalize(vmin=0, vmax=1.2)

    rgba = cmap(norm(0))
    print(mpl.colors.to_hex(rgba))
    rgba = cmap(norm(1))
    print(mpl.colors.to_hex(rgba))
