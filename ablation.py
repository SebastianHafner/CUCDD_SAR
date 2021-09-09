from tqdm import tqdm
import numpy as np
from utils import dataset_helpers, config, label_helpers, metrics, geofiles
import change_detection_models as cd_models
import matplotlib.pyplot as plt


def quanitative_evaluation(model: cd_models.ChangeDetectionMethod) -> tuple:
    preds, gts = [], []
    for aoi_id in dataset_helpers.get_aoi_ids():
        if dataset_helpers.length_timeseries(aoi_id, config.include_masked()) > 6:
            pred = model.change_detection(aoi_id)
            preds.append(pred.flatten())
            gt = label_helpers.generate_change_label(aoi_id, config.include_masked())
            gts.append(gt.flatten())
            assert (pred.size == gt.size)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    precision = metrics.compute_precision(preds, gts)
    recall = metrics.compute_recall(preds, gts)
    f1 = metrics.compute_f1_score(preds, gts)

    return f1, precision, recall


def ablation1(error_multiplier: int, min_diff_range: tuple, step_size: float, band: str = 'VV'):

    min_diff_start, min_diff_end = min_diff_range
    min_diff_candidates = np.arange(min_diff_start, min_diff_end + step_size, step_size)
    file = config.root_path() / 'ablation' / f'ablation1_{band}_{error_multiplier}.json'

    if file.exists():
        ablation_data = geofiles.load_json(file)
    else:
        ablation_data = {
            'min_diff_range': min_diff_range,
            'step_size': step_size,
            'f1_score': [],
            'precision': [],
            'recall': []
        }

        for min_diff_candidate in tqdm(min_diff_candidates):
            sf = cd_models.StepFunctionModel(band, error_multiplier=error_multiplier, min_diff=min_diff_candidate,
                                             min_segment_length=2)
            f1, precision, recall = quanitative_evaluation(sf)
            ablation_data['f1_score'].append(f1)
            ablation_data['precision'].append(precision)
            ablation_data['recall'].append(recall)
        geofiles.write_json(file, ablation_data)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(min_diff_candidates, ablation_data['f1_score'], label='F1 score')
    ax.plot(min_diff_candidates, ablation_data['precision'], label='Precision')
    ax.plot(min_diff_candidates, ablation_data['recall'], label='Recall')

    ax.set_xlim([min_diff_start, min_diff_end])
    ax.set_ylim([0, 1])
    plt.legend()
    plt.show()


def ablation2(min_diff: float, error_multiplier_range: tuple, step_size: float, band: str = 'VV'):
    error_multiplier_start, error_multiplier_end = error_multiplier_range
    error_multiplier_candidates = np.arange(error_multiplier_start, error_multiplier_end + step_size, step_size)
    file = config.root_path() / 'ablation' / f'ablation2_{band}_{min_diff:.1f}.json'

    if file.exists():
        ablation_data = geofiles.load_json(file)
    else:
        ablation_data = {
            'error_multiplier_range': error_multiplier_range,
            'step_size': step_size,
            'f1_score': [],
            'precision': [],
            'recall': []
        }

        for error_multiplier_candidate in tqdm(error_multiplier_candidates):
            sf = cd_models.StepFunctionModel(band, error_multiplier=error_multiplier_candidate, min_diff=min_diff,
                                             min_segment_length=2)
            f1, precision, recall = quanitative_evaluation(sf)
            ablation_data['f1_score'].append(f1)
            ablation_data['precision'].append(precision)
            ablation_data['recall'].append(recall)
        geofiles.write_json(file, ablation_data)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(error_multiplier_candidates, ablation_data['f1_score'], label='F1 score')
    ax.plot(error_multiplier_candidates, ablation_data['precision'], label='Precision')
    ax.plot(error_multiplier_candidates, ablation_data['recall'], label='Recall')

    ax.set_xlim([error_multiplier_start, error_multiplier_end])
    ax.set_ylim([0, 1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # ablation1(3, min_diff_range=(0, 10), step_size=1)
    ablation2(2, error_multiplier_range=(0, 5), step_size=0.5)


