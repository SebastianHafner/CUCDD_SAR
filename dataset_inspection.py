from pathlib import Path
from utils import geofiles, visualization, dataset_helpers, label_helpers, metrics, mask_helpers, config
from utils import input_helpers
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def visualize_satellite_data(dataset: str, aoi_id: str, save_plot: bool = False):
    timestamps = dataset_helpers.timestamps(dataset)
    ts = timestamps[aoi_id]
    n = len(ts)
    n_rows = 4
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n, figsize=(n * plot_size, n_rows * plot_size))

    for i, (year, month, _) in enumerate(tqdm(ts)):
        visualization.plot_optical(axs[0, i], dataset, aoi_id, year, month, vis='true_color')
        visualization.plot_optical(axs[1, i], dataset, aoi_id, year, month, vis='false_color')
        visualization.plot_sar(axs[2, i], dataset, aoi_id, year, month, vis='VV')
        visualization.plot_sar(axs[3, i], dataset, aoi_id, year, month, vis='VH')

        title = f'{i} {year}-{month:02d}'

        axs[0, i].set_title(title, c='k', fontsize=16, fontweight='bold')

    if not save_plot:
        plt.show()
    else:
        dataset_name = dataset_helpers.dataset_name(dataset)
        output_file = dataset_helpers.root_path() / 'plots' / 'inspection' / dataset_name / f'{aoi_id}_satellite.png'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_all_data(dataset: str, aoi_id: str, save_plot: bool = False):

    metadata = dataset_helpers.metadata(dataset)
    ts = metadata['aois'][aoi_id]
    n = len(ts)
    gt_available = True if dataset == 'spacenet7' else False
    n_rows = 4 if gt_available else 3
    plot_size = 3

    fig, axs = plt.subplots(n_rows, n, figsize=(n * plot_size, n_rows * plot_size))

    for i, (year, month, mask, s1, s2) in enumerate(tqdm(ts)):
        visualization.plot_sar(axs[0, i], ds, aoi_id, year, month)
        visualization.plot_optical(axs[1, i], ds, aoi_id, year, month)

        if gt_available:
            visualization.plot_buildings(axs[2, i], aoi_id, year, month)

        # title
        title = f'{i} {year}-{month:02d}'
        if mask:
            color = 'blue'
        else:
            if s1 and s2:
                color = 'green'
            elif s1:
                color = 'orange'
            elif s2:
                color = 'cyan'
            else:
                color = 'red'
        if gt_available:
            fully_masked = mask_helpers.is_fully_masked(dataset, aoi_id, year, month)
            prediction_available = input_helpers.prediction_is_available(dataset, aoi_id, year, month)
            if fully_masked or not prediction_available:
                f1 = 'NaN'
            else:
                y_true = label_helpers.load_label(aoi_id, year, month)
                y_pred = input_helpers.load_prediction(dataset, aoi_id, year, month)
                y_pred = y_pred > 0.5
                f1 = metrics.compute_f1_score(y_pred.flatten(), y_true.flatten())
                f1 = f'{f1*100:.1f}'
            title = f'{title} {f1}'

        axs[0, i].set_title(title, c=color, fontsize=12, fontweight='bold')
        visualization.plot_prediction(axs[n_rows - 1, i], dataset, aoi_id, year, month)

    if not save_plot:
        plt.show()
    else:
        dataset_name = dataset_helpers.dataset_name(dataset)
        config_name = dataset_helpers.config_name()
        output_path = dataset_helpers.root_path() / 'plots' / 'inspection' / dataset_name / config_name
        output_file = output_path / f'{aoi_id}_data.png'
        output_file.parent.mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_timeseries_length(dataset: str, sort_by_length: bool = False, numeric_names: bool = False):
    data, labels = [], []
    aoi_ids = dataset_helpers.get_aoi_ids(dataset)
    sensor = dataset_helpers.settings()['INPUT']['SENSOR']
    for aoi_id in tqdm(aoi_ids):
        if aoi_id in dataset_helpers.missing_aois():
            continue
        metadata = dataset_helpers.metadata(dataset)['aois'][aoi_id]
        if sensor == 'sentinel1':
            metadata = [mask for _, _, mask, s1, s2 in metadata if s1]
        elif sensor == 'sentinel2':
            metadata = [mask for _, _, mask, s1, s2 in metadata if s2]
        else:
            metadata = [mask for _, _, mask, s1, s2 in metadata if s1 and s2]

        n_clear = len([mask for mask in metadata if not mask])
        n_clear_masked = len([mask for mask in metadata if mask])
        data.append([n_clear, n_clear_masked])
        labels.append(aoi_id)

    if sort_by_length:
        data = sorted(data, key=lambda d: d[0])
    clear = [d[0] for d in data]
    clear_masked = [d[1] for d in data]
    if numeric_names:
        labels = [f'AOI {i + 1}' for i in range(len(labels))]
    else:
        labels = [aoi_id[4:15] for aoi_id in labels]
    print(data, labels)
    width = 0.5  # the width of the bars: can also be len(x) sequence

    fontsize = 20
    fig, ax = plt.subplots(1, 1, figsize=(len(labels) / 2, 6))

    ax.bar(labels, clear, width, label='Unmasked')
    ax.bar(labels, clear_masked, width, bottom=clear, label='Masked')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
    ax.set_xlim((-0.5, len(labels) - 0.5))

    max_value = (max([c + cm for c, cm in data]) // 5 + 1) * 5
    y_ticks = np.arange(0, max_value + 1, 5)
    ax.set_ylim((0, max_value))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y_tick:.0f}' for y_tick in y_ticks], fontsize=fontsize)
    ax.set_ylabel('Timeseries length', fontsize=fontsize)

    ax.legend(ncol=2, handletextpad=0.4, columnspacing=1.2, frameon=False, loc='upper center', fontsize=fontsize)

    plt.show()


def produce_cnn_timeseries_cube(dataset: str, aoi_id: str):
    probs = input_helpers.load_input_timeseries(dataset, aoi_id, dataset_helpers.include_masked())
    probs = (probs * 100).astype(np.uint8)
    transform, crs = dataset_helpers.get_geo(dataset, aoi_id)
    input_name = input_helpers.input_name()
    file = dataset_helpers.root_path() / 'inspection' / f'{input_name}_{aoi_id}.tif'
    geofiles.write_tif(file, probs, transform, crs)


def produce_satellite_timeseries_cube(dataset: str, aoi_id: str, satellite: str, band: str):
    data = input_helpers.load_satellite_timeseries(dataset, aoi_id, satellite, band)
    transform, crs = dataset_helpers.get_geo(dataset, aoi_id)
    data_name = f'{satellite}_{band}'
    save_path = dataset_helpers.root_path() / 'inspection' / data_name
    save_path.mkdir(exist_ok=True)
    file = save_path / f'{data_name}_{aoi_id}.tif'
    geofiles.write_tif(file, data.astype(np.float32), transform, crs)


def produce_change_date_label(dataset: str, aoi_id: str):
    change_date = label_helpers.generate_change_date_label(aoi_id, dataset_helpers.include_masked())
    transform, crs = dataset_helpers.get_geo(dataset, aoi_id)
    save_path = dataset_helpers.root_path() / 'inspection' / 'change_date'
    save_path.mkdir(exist_ok=True)
    file = save_path / f'change_date_{aoi_id}.tif'
    geofiles.write_tif(file, change_date.astype(np.uint8), transform, crs)


def show_data_availability(dataset: str, aoi_id: str):
    timeseries = dataset_helpers.aoi_metadata(dataset, aoi_id)
    n = len(timeseries)

    fig, ax = plt.subplots(1, 1, figsize=(n, 5))
    fontsize = 20

    coords = [[], []]
    for index, (_, _, mask, s1, s2) in enumerate(timeseries):
        if s1:
            coords[0].append(index)
            coords[1].append(1)
        if s2:
            coords[0].append(index)
            coords[1].append(2)
        if not mask:
            coords[0].append(index)
            coords[1].append(3)

    ax.scatter(*coords, marker='+', c='k', s=100)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_title(aoi_id, fontsize=fontsize)

    # y axis
    ax.set_ylim((0.5, 3.5))
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['S1', 'S2', 'Planet'], fontsize=fontsize)

    # x axis
    x_labels = [f'{str(year)[2:]}-{month:02d}' for year, month, *_ in timeseries]
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(x_labels, fontsize=fontsize)

    plt.show()


def study_site_mosaic(dataset: str, satellite: str, grid: np.ndarray = None, n_cols: int = 5):
    aoi_ids = dataset_helpers.get_aoi_ids(dataset)

    if grid is None:
        n_rows = len(aoi_ids) // n_cols
        if len(aoi_ids) % n_cols != 0:
            n_rows += 1
        grid = np.ones((n_rows, n_cols))
    n_rows, n_cols = grid.shape

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * config.plotsize(), n_rows * config.plotsize()))

    aoi_index = 0
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i, j]

            if aoi_index < len(aoi_ids) and bool(grid[i, j]):
                aoi_id = aoi_ids[aoi_index]

                year, month = dataset_helpers.get_date_from_index(0, dataset, aoi_id, config.include_masked())
                if satellite == 'sentinel1':
                    visualization.plot_sar(ax, dataset, aoi_id, year, month)
                else:
                    visualization.plot_optical(ax, dataset, aoi_id, year, month)
                ax.set_title(f'AOI {aoi_index + 1}', fontsize=config.fontsize())
                aoi_index += 1
            else:
                ax.axis('off')

    file = config.root_path() / 'plots' / 'inspection' / 'study_sites_mosaic.png'
    plt.savefig(file, dpi=300, bbox_inches='tight')
    plt.show()


def print_dataset_size(dataset: str):
    n = 0
    for aoi_id in dataset_helpers.get_aoi_ids(dataset):
        n += dataset_helpers.length_timeseries(dataset, aoi_id, config.include_masked())
    print(n)


if __name__ == '__main__':
    ds = 'spacenet7'
    for i, aoi_id in enumerate(dataset_helpers.get_aoi_ids(ds)):
        # produce_satellite_timeseries_cube(ds, aoi_id, 'sentinel1', 'VV')
        # produce_change_date_label(ds, aoi_id)
        # visualize_satellite_data(ds, aoi_id, save_plot=True)
        # show_data_availability(ds, aoi_id)
        # visualize_all_data(ds, aoi_id, save_plot=True)
        # visualize_timeseries(ds, aoi_id, config_name=cfg, save_plot=True)
        # produce_timeseries_cube(ds, aoi_id)
        pass

    # grid = np.array([[0, 0, 0, 0, 1],
    #                  [0, 0, 0, 0, 1],
    #                  [0, 0, 0, 0, 1],
    #                  [1, 1, 1, 1, 1],
    #                  [1, 1, 1, 1, 1]])
    #
    # study_site_mosaic(ds, 'sentinel2', grid=grid)
    # visualize_timeseries_length(ds, numeric_names=True)
    print_dataset_size(ds)
