from utils import dataset_helpers, label_helpers, visualization, metrics, geofiles, config, sentinel1_helpers
from skimage.filters import threshold_otsu, threshold_local, threshold_minimum
import matplotlib.pyplot as plt
import change_detection_models as cd_models
import numpy as np
from tqdm import tqdm
FONTSIZE = 16


def qualitative_testing(model: cd_models.ChangeDetectionMethod, aoi_id: str, save_plot: bool = False):

    dates = dataset_helpers.get_timeseries(aoi_id)
    start_year, start_month, *_ = dates[0]
    end_year, end_month, *_ = dates[-1]

    fig, axs = plt.subplots(1, 6, figsize=(20, 5))
    plt.tight_layout()

    # pre image, post image and gt
    visualization.plot_planet_monthly_mosaic(axs[0], aoi_id, start_year, start_month)
    axs[0].set_title('Planet t1', fontsize=FONTSIZE)
    visualization.plot_planet_monthly_mosaic(axs[1], aoi_id, end_year, end_month)
    axs[1].set_title('Planet tn', fontsize=FONTSIZE)
    visualization.plot_sar(axs[2], aoi_id, start_year, start_month)
    axs[2].set_title('SAR t1', fontsize=FONTSIZE)
    visualization.plot_sar(axs[3], aoi_id, end_year, end_month)
    axs[3].set_title('SAR t2', fontsize=FONTSIZE)

    visualization.plot_change_label(axs[4], aoi_id)
    axs[4].set_title('Change GT', fontsize=FONTSIZE)

    change = model.change_detection(aoi_id)
    visualization.plot_blackwhite(axs[5], change)
    axs[5].set_title('Change Pred', fontsize=FONTSIZE)

    if not save_plot:
        plt.show()
    else:
        save_path = config.output_path() / 'plots' / 'testing' / model.name
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'change_{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def qualitative_testing_v2(model: cd_models.ChangeDetectionMethod, aoi_id: str, save_plot: bool = False):
    dates = dataset_helpers.get_timeseries(aoi_id)
    start_year, start_month, *_ = dates[0]
    end_year, end_month, *_ = dates[-1]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    plt.tight_layout()
    for _, ax in np.ndenumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])

    # pre image, post image and gt
    visualization.plot_sar(axs[0, 0], aoi_id, start_year, start_month)
    axs[0, 0].set_xlabel('(a) SAR t1', fontsize=FONTSIZE)
    visualization.plot_sar(axs[0, 1], aoi_id, end_year, end_month)
    axs[0, 1].set_xlabel('(b) SAR t2', fontsize=FONTSIZE)

    sar_t1 = sentinel1_helpers.load_sentinel1_band(aoi_id, start_year, start_month, 'VV')
    sar_t2 = sentinel1_helpers.load_sentinel1_band(aoi_id, end_year, end_month, 'VV')
    log_ratio = sar_t2 - sar_t1
    axs[0, 2].imshow(log_ratio)
    axs[0, 2].set_xlabel('(c) SAR log-ratio', fontsize=FONTSIZE)

    visualization.plot_change_label(axs[1, 0], aoi_id)
    axs[1, 0].set_xlabel('(d) Ground truth', fontsize=FONTSIZE)

    block_size = 25
    # thresh = threshold_local(log_ratio, block_size)
    thresh = threshold_otsu(log_ratio)
    change_log_ratio = log_ratio > thresh
    axs[1, 1].set_xlabel('(e) Predicted (log-ratio)', fontsize=FONTSIZE)
    visualization.plot_blackwhite(axs[1, 1], change_log_ratio)

    change = model.change_detection(aoi_id)
    visualization.plot_blackwhite(axs[1, 2], change)
    axs[1, 2].set_xlabel('(f) Predicted (proposed)', fontsize=FONTSIZE)

    if not save_plot:
        plt.show()
    else:
        save_path = config.output_path() / 'plots' / 'testing' / model.name
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'change_{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def quantitative_testing(model: cd_models.ChangeDetectionMethod, aoi_id: str) -> tuple:

    pred = model.change_detection(aoi_id)
    gt = label_helpers.generate_change_label(aoi_id)

    precision = metrics.compute_precision(pred, gt)
    recall = metrics.compute_recall(pred, gt)
    f1 = metrics.compute_f1_score(pred, gt)

    return f1, precision, recall


def show_quantitative_testing(model: cd_models.ChangeDetectionMethod, aoi_id: str):
    f1, precision, recall = quantitative_testing(model, aoi_id)
    print(aoi_id)
    print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


def quantitative_testing_dataset(model: cd_models.ChangeDetectionMethod):
    preds, gts = [], []
    for aoi_id in tqdm(dataset_helpers.get_aoi_ids()):
        if dataset_helpers.length_timeseries(aoi_id) > 6:
            pred = model.change_detection(aoi_id)
            preds.append(pred.flatten())
            gt = label_helpers.generate_change_label(aoi_id)
            gts.append(gt.flatten())
            assert(pred.size == gt.size)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    precision = metrics.compute_precision(preds, gts)
    recall = metrics.compute_recall(preds, gts)
    f1 = metrics.compute_f1_score(preds, gts)

    print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


def run_change_detection_inference(model: cd_models.ChangeDetectionMethod):
    for aoi_id in tqdm(dataset_helpers.get_aoi_ids()):
        pred = model.change_detection(aoi_id)
        transform, crs = dataset_helpers.get_geo(aoi_id)
        path = config.root_path() / 'inference' / model.name / config.config_name()
        path.mkdir(exist_ok=True)
        file = path / f'change_{aoi_id}.tif'
        geofiles.write_tif(file, pred.astype(np.uint8), transform, crs)


def qualitative_testing_assembled(model: cd_models.ChangeDetectionMethod, aoi_ids: list, save_plot: bool = True):
    plot_size = 3
    fig, axs = plt.subplots(len(aoi_ids), 6, figsize=(6 * plot_size, len(aoi_ids) * plot_size), tight_layout=True)

    vertical_label_offset = -0.05
    axs[-1, 0].set_xlabel(r'(a) Planet $t_1$', fontsize=FONTSIZE)
    axs[-1, 0].xaxis.set_label_coords(0.5, vertical_label_offset)
    axs[-1, 1].set_xlabel(r'(b) Planet $t_n$', fontsize=FONTSIZE)
    axs[-1, 1].xaxis.set_label_coords(0.5, vertical_label_offset)
    axs[-1, 2].set_xlabel(r'(c) SAR $t_1$', fontsize=FONTSIZE)
    axs[-1, 2].xaxis.set_label_coords(0.5, vertical_label_offset)
    axs[-1, 3].set_xlabel(r'(d) SAR $t_n$', fontsize=FONTSIZE)
    axs[-1, 3].xaxis.set_label_coords(0.5, vertical_label_offset)
    axs[-1, 4].set_xlabel(r'(e) Change GT', fontsize=FONTSIZE)
    axs[-1, 4].xaxis.set_label_coords(0.5, vertical_label_offset)
    axs[-1, 5].set_xlabel(r'(f) Change Pred', fontsize=FONTSIZE)
    axs[-1, 5].xaxis.set_label_coords(0.5, vertical_label_offset)

    for i, aoi_id in enumerate(aoi_ids):
        dates = dataset_helpers.get_timeseries(aoi_id)
        start_year, start_month, *_ = dates[0]
        end_year, end_month, *_ = dates[-1]

        visualization.plot_planet_monthly_mosaic(axs[i, 0], aoi_id, start_year, start_month)
        visualization.plot_planet_monthly_mosaic(axs[i, 1], aoi_id, end_year, end_month)
        visualization.plot_sar(axs[i, 2], aoi_id, start_year, start_month)
        visualization.plot_sar(axs[i, 3], aoi_id, end_year, end_month)
        visualization.plot_change_label(axs[i, 4], aoi_id)
        change = model.change_detection(aoi_id)
        visualization.plot_blackwhite(axs[i, 5], change)

    if not save_plot:
        plt.show()
    else:
        save_path = config.output_path() / 'plots' / 'testing' / model.name
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'change_assembled_{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def box_plots(model: cd_models.ChangeDetectionMethod, save_plot: bool = False):


    f1s, precisions, recalls = [], [], []
    for i, aoi_id in enumerate(tqdm(dataset_helpers.get_aoi_ids(min_timeseries_length=config.min_timeseries_length()))):
        f1, precision, recall = quantitative_testing(model, aoi_id)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        if i > 1:
            pass

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    data = [recalls, precisions, f1s]
    y_ticks = np.arange(3)
    ax.boxplot(data, positions=y_ticks, whis=(5, 95), widths=0.3, vert=False)
    y_tick_labels = ['Recall', 'Precision', 'F1 score']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, rotation=0, fontsize=FONTSIZE)
    x_ticks = np.linspace(0, 1, 5)
    ax.set_xlim((0, 1))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{x_tick:.1f}' for x_tick in x_ticks], fontsize=FONTSIZE)

    if not save_plot:
        plt.show()
    else:
        save_path = config.output_path() / 'plots' / 'testing' / model.name
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'box_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    # sf = cd_models.StepFunctionModel('VV', error_multiplier=2, min_diff=5, min_segment_length=2, noise_reduction=True)
    model = cd_models.SimpleStepFunctionModel('VV', min_diff=5)
    model = cd_models.LogRatioModel('VV')
    for i, aoi_id in enumerate(tqdm(dataset_helpers.get_aoi_ids(min_timeseries_length=config.min_timeseries_length()))):
        # qualitative_testing_v2(model, aoi_id, save_plot=False)
        # quantitative_testing(model, aoi_id)
        pass

    # qualitative_testing(model, ds, 'L15-0566E-1185N_2265_3451_13', save_plot=False)

    # quantitative_testing_dataset(sf)
    # quantitative_testing(model, ds, 'L15-0683E-1006N_2732_4164_13')
    # run_change_detection_inference(sf, ds)

    aoi_ids = [
        'L15-0357E-1223N_1429_3296_13',
        'L15-0358E-1220N_1433_3310_13',
        'L15-1204E-1204N_4819_3372_13',
        'L15-1296E-1198N_5184_3399_13',
        'L15-0614E-0946N_2459_4406_13',
        'L15-0683E-1006N_2732_4164_13',
        'L15-0924E-1108N_3699_3757_13',
        'L15-1049E-1370N_4196_2710_13',
    ]
    qualitative_testing_assembled(model, aoi_ids, False)
    # box_plots(sf, save_plot=True)
