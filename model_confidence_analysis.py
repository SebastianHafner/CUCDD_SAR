from utils import dataset_helpers, label_helpers, visualization, metrics
import matplotlib.pyplot as plt
import change_detection_models as cd_models
import numpy as np
from tqdm import tqdm


def plot_confidence_f1_correlation(model: cd_models.StepFunctionModel, dataset: str, include_masked_data: bool = False,
                                   n_bins: int = 2):
    preds, gts, confidence = [], [], []
    for i, aoi_id in enumerate(tqdm(dataset_helpers.get_aoi_ids(dataset))):
        pred = model.change_detection(dataset, aoi_id, include_masked_data)
        preds.append(pred.flatten())
        conf = model.model_confidence(dataset, aoi_id, include_masked_data)
        confidence.append(conf.flatten())
        gt = label_helpers.generate_change_label(dataset, aoi_id, include_masked_data)
        gts.append(gt.flatten())

        assert(pred.size == gt.size == conf.size)
        if i == 5:
            pass

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    confidence = np.concatenate(confidence)

    bins = list(np.linspace(0, 1, n_bins + 1))
    for i in range(1, len(bins)):
        bottom_value = bins[i - 1]
        top_value = bins[i]

        in_bin = np.logical_and(confidence > bottom_value, confidence <= top_value)
        n_sub = np.sum(in_bin)
        preds_sub = preds[in_bin]
        gts_sub = gts[in_bin]

        precision = metrics.compute_precision(preds_sub, gts_sub)
        recall = metrics.compute_recall(preds_sub, gts_sub)
        f1 = metrics.compute_f1_score(preds_sub, gts_sub)

        print(f'Bin ({bottom_value:.2f}, {top_value:.2f}] ({n_sub})')
        print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


if __name__ == '__main__':
    ds = 'spacenet7'
    model = cd_models.StepFunctionModel(n_stable=6)
    plot_confidence_f1_correlation(model, ds, include_masked_data=True, n_bins=5)
