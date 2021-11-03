import numpy as np
from tqdm import tqdm
from utils import dataset_helpers, config, label_helpers, metrics, geofiles
import change_detection_models as cd_models
import matplotlib.pyplot as plt

FONTSIZE = 16


def quanitative_evaluation(model: cd_models.ChangeDetectionMethod) -> tuple:
    preds, gts = [], []
    for aoi_id in dataset_helpers.get_aoi_ids(min_timeseries_length=config.min_timeseries_length()):
        pred = model.change_detection(aoi_id)
        preds.append(pred.flatten())
        gt = label_helpers.generate_change_label(aoi_id)
        gts.append(gt.flatten())
        assert (pred.size == gt.size)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    precision = metrics.compute_precision(preds, gts)
    recall = metrics.compute_recall(preds, gts)
    f1 = metrics.compute_f1_score(preds, gts)

    return f1, precision, recall


# em: error multiplier, mdr: min diff probability
def run_grid_search(em_range: tuple, em_step_size: int, mdb_range: tuple, mdb_step_size: float,
                    force_run: bool = False):
    em_start, em_end = em_range
    em_candidates = np.arange(em_start, em_end + em_step_size, em_step_size)
    m = len(em_candidates)

    mdb_start, mdb_end = mdb_range
    mdb_candidates = np.arange(mdb_start, mdb_end + mdb_step_size, mdb_step_size)
    n = len(mdb_candidates)

    fname = f'grid_search_{config.subset_activated()}.json'
    file = config.output_path() / 'grid_search' / fname

    if file.exists() and not force_run:
        ablation_data = geofiles.load_json(file)
    else:
        ablation_data = {
            'em_range': em_range,
            'em_step_size': em_step_size,
            'mdb_range': mdb_range,
            'mdb_step_size': mdb_step_size,
            'data': []
        }

        for i, em_candidate in enumerate(em_candidates):
            for j, mdb_candidate in enumerate(tqdm(mdb_candidates)):
                model = cd_models.StepFunctionModel('VV', error_multiplier=em_candidate, min_diff=mdb_candidate,
                                                    min_segment_length=2, noise_reduction=True)
                f1, precision, recall = quanitative_evaluation(model)
                ablation_data['data'].append({
                    'index': (i, j),
                    'em': int(em_candidate),
                    'mdb': float(mdb_candidate),
                    'f1_score': float(f1),
                    'precision': float(precision),
                    'recall': float(recall),
                })
        geofiles.write_json(file, ablation_data)

    acc_metrics = ['f1_score', 'precision', 'recall']

    for metric in acc_metrics:

        matrix = np.empty((m, n), dtype=np.single)
        for d in ablation_data['data']:
            i, j = d['index']
            matrix[i, j] = d[metric]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        vmax = 0.5
        img = ax.imshow(matrix, vmin=0, vmax=vmax, cmap='jet')
        xticks = np.arange(0, n, 2)
        mdp_ticks = np.arange(mdb_start, mdb_end + mdb_step_size, mdb_step_size * 2)
        xticklabels = [f'{mdp_tick:.1f}' for mdp_tick in mdp_ticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=FONTSIZE)
        ax.set_xlabel(r'$\lambda_2$ (min probability increase)', fontsize=FONTSIZE)
        yticks = np.arange(m)
        yticklabels = [f'{cand:.0f}' for cand in em_candidates]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=FONTSIZE)
        ax.set_ylabel(r'$\lambda_1$ (error multiplier)', fontsize=FONTSIZE)
        cbar = fig.colorbar(img, ax=ax)
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel(metric, rotation=270, fontsize=FONTSIZE)
        cbartick_stepsize = 0.1
        cbarticks = np.arange(0, vmax + cbartick_stepsize, cbartick_stepsize)
        cbar_ticklabels = [f'{cbartick:.1f}' for cbartick in cbarticks]
        cbar.ax.get_yaxis().set_ticks(cbarticks)
        cbar.ax.get_yaxis().set_ticklabels(cbar_ticklabels, fontsize=FONTSIZE)
        plt.show()
        plt.close(fig)

        print(f'{metric}: {np.max(matrix):.3f} ({np.argmax(matrix)})')


if __name__ == '__main__':
    run_grid_search(em_range=(1, 6), em_step_size=1, mdb_range=(1, 10), mdb_step_size=1, force_run=False)

