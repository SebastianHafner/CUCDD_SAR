from utils import dataset_helpers, label_helpers, visualization, metrics, geofiles, config
import matplotlib.pyplot as plt
import change_detection_models as cd_models
import numpy as np
from tqdm import tqdm


def qualitative_testing(model: cd_models.ChangeDetectionMethod, aoi_id: str, save_plot: bool = False,
                        color_misclassifications: bool = False, sensor: str = 'sentinel2', show_f1: bool = True):

    dates = dataset_helpers.get_timeseries(aoi_id, config.include_masked())
    start_year, start_month, *_ = dates[0]
    end_year, end_month, *_ = dates[-1]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # pre image, post image and gt
    if sensor == 'sentinel2':
        visualization.plot_optical(axs[0], aoi_id, start_year, start_month)
        visualization.plot_optical(axs[1], aoi_id, end_year, end_month)
    else:
        visualization.plot_sar(axs[0], aoi_id, start_year, start_month)
        visualization.plot_sar(axs[1], aoi_id, end_year, end_month)

    axs[0].set_title('S2 Start TS')
    axs[1].set_title('S2 End TS')

    visualization.plot_change_label(axs[2], aoi_id, config.include_masked())
    axs[2].set_title('Change GT')

    change = model.change_detection(aoi_id)
    if color_misclassifications:
        visualization.plot_classification(axs[3], change, aoi_id)
    else:
        visualization.plot_blackwhite(axs[3], change)
    title = 'Change Pred'
    if show_f1:
        label = label_helpers.generate_change_label(aoi_id, config.include_masked())
        f1 = metrics.compute_f1_score(change.flatten(), label.flatten())
        title = f'{title} (F1 {f1:.3f})'
    axs[3].set_title(title)

    if not save_plot:
        plt.show()
    else:
        save_path = config.root_path() / 'plots' / 'testing' / model.name
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'change_{aoi_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def quantitative_testing(model: cd_models.ChangeDetectionMethod, dataset: str, aoi_id: str):

    pred = model.change_detection(aoi_id)
    gt = label_helpers.generate_change_label(aoi_id)

    precision = metrics.compute_precision(pred, gt)
    recall = metrics.compute_recall(pred, gt)
    f1 = metrics.compute_f1_score(pred, gt)

    print(aoi_id)
    print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


def quantitative_testing_dataset(model: cd_models.ChangeDetectionMethod):
    preds, gts = [], []
    for aoi_id in tqdm(dataset_helpers.get_aoi_ids()):
        if dataset_helpers.length_timeseries(aoi_id, config.include_masked()) > 6:
            pred = model.change_detection(aoi_id)
            preds.append(pred.flatten())
            gt = label_helpers.generate_change_label(aoi_id, config.include_masked())
            gts.append(gt.flatten())
            assert(pred.size == gt.size)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    precision = metrics.compute_precision(preds, gts)
    recall = metrics.compute_recall(preds, gts)
    f1 = metrics.compute_f1_score(preds, gts)

    print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


def run_change_detection_inference(model: cd_models.ChangeDetectionMethod, dataset: str):
    for aoi_id in tqdm(dataset_helpers.get_aoi_ids()):
        pred = model.change_detection(dataset, aoi_id)
        transform, crs = dataset_helpers.get_geo(dataset, aoi_id)
        path = config.root_path() / 'inference' / model.name / config.config_name()
        path.mkdir(exist_ok=True)
        file = path / f'change_{aoi_id}.tif'
        geofiles.write_tif(file, pred.astype(np.uint8), transform, crs)


if __name__ == '__main__':
    sf = cd_models.StepFunctionModel('VV', error_multiplier=3, min_diff=0.2, min_segment_length=2)
    for i, aoi_id in enumerate(tqdm(dataset_helpers.get_aoi_ids())):
        if dataset_helpers.length_timeseries(aoi_id, config.include_masked()) > 6:
            qualitative_testing(sf, aoi_id, save_plot=False, sensor='sentinel2')
            # quantitative_testing(model, ds, aoi_id)
            pass

    # qualitative_testing(model, ds, 'L15-0566E-1185N_2265_3451_13', save_plot=False)

    quantitative_testing_dataset(sf)
    # quantitative_testing(model, ds, 'L15-0683E-1006N_2732_4164_13')
    # run_change_detection_inference(sf, ds)
