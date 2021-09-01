from utils import dataset_helpers, label_helpers, input_helpers, metrics, config
import matplotlib.pyplot as plt
import change_detection_models as cd_models
import numpy as np
from tqdm import tqdm
import scipy

FONTSIZE = 20


# supported modes: first_last, all
def sensitivity_to_f1(model: cd_models.ChangeDetectionMethod, mode: str, include_regression_line: bool = False,
                      save_plot: bool = False):

    f1_scores_extraction = []
    f1_scores_change = []

    for i, aoi_id in enumerate(tqdm(dataset_helpers.get_aoi_ids('spacenet7'))):

        # compute average urban extraction f1 score
        f1_scores_ts = []
        ts = dataset_helpers.get_timeseries('spacenet7', aoi_id, config.include_masked())
        if mode == 'first_last':
            ts = [ts[0], ts[-1]]
        for year, month, *_ in ts:
            label = label_helpers.load_label(aoi_id, year, month)
            pred = input_helpers.load_prediction('spacenet7', aoi_id, year, month)
            pred = pred > 0.5
            f1_scores_ts.append(metrics.compute_f1_score(pred, label))
        f1_scores_extraction.append(np.mean(f1_scores_ts))

        # compute change f1 score
        pred_change = model.change_detection('spacenet7', aoi_id)
        label_change = label_helpers.generate_change_label('spacenet7', aoi_id, config.include_masked())
        f1_scores_change.append(metrics.compute_f1_score(pred_change, label_change))

    fig, ax = plt.subplots(figsize=(10, 10))

    color = '#1f77b4' if config.input_sensor() == 'sentinel1' else '#d62728'
    ax.scatter(f1_scores_extraction, f1_scores_change, c=color)
    ticks = np.linspace(0, 1, 6)
    tick_labels = [f'{tick:.1f}' for tick in ticks]
    ax.set_xlim((0, 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=FONTSIZE)
    ax.set_xlabel('F1 score urban extraction', fontsize=FONTSIZE)
    ax.set_ylim((0, 1))
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=FONTSIZE)
    ax.set_ylabel('F1 score change detection', fontsize=FONTSIZE)

    if include_regression_line:
        linreg = scipy.stats.linregress(f1_scores_extraction, f1_scores_change)
        values = np.linspace(0, 1, 5)
        label = f'{linreg.intercept:.2f} + {linreg.slope:.2f}x (R-square = {linreg.rvalue:.2f})'
        ax.plot(values, linreg.intercept + linreg.slope * values, 'k', label=label)
        ax.legend(frameon=False, loc='upper left', fontsize=FONTSIZE)

    if not save_plot:
        plt.show()
    else:
        save_path = config.root_path() / 'plots' / 'sensitivity_analysis'
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'{model.name}_{mode}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def sensitivity_to_omissions(model: cd_models.ChangeDetectionMethod, include_masked_data: bool = False,
                             save_plot: bool = False):

    omissions_extraction = []
    f1_scores_change = []

    for i, aoi_id in enumerate(tqdm(dataset_helpers.get_aoi_ids('spacenet7'))):

        # compute average urban extraction f1 score
        omissions = input_helpers.compute_omissions('spacenet7', aoi_id, include_masked_data)
        omission_rate = np.sum(omissions) / np.size(omissions)
        omissions_extraction.append(omission_rate)

        # compute change f1 score
        pred_change = model.change_detection('spacenet7', aoi_id, include_masked_data)
        label_change = label_helpers.generate_change_label('spacenet7', aoi_id, include_masked_data)
        f1_scores_change.append(metrics.compute_f1_score(pred_change, label_change))

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.scatter(omissions_extraction, f1_scores_change, c='k')
    y_ticks = np.linspace(0, 1, 6)
    y_tick_labels = [f'{tick:.1f}' for tick in y_ticks]
    ticks = [3, 2.5, 2, 1.5]
    x_ticklabels = [f'{x_tick:.1f}' for x_tick in ticks]
    x_ticks = [10**-x for x in ticks]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=FONTSIZE)
    ax.set_xlabel('Omission rate urban extraction (2e-x %)', fontsize=FONTSIZE)
    ax.set_ylim((0, 1))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=FONTSIZE)
    ax.set_ylabel('F1 score change detection', fontsize=FONTSIZE)

    if not save_plot:
        plt.show()
    else:
        save_path = dataset_helpers.root_path() / 'plots' / 'sensitivity_analysis'
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'{model.name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':

    sf = cd_models.StepFunctionModel(error_multiplier=3, min_prob_diff=0.2, min_segment_length=2)
    sensitivity_to_f1(sf, 'all', include_regression_line=True, save_plot=False)
    # sensitivity_to_omissions(stepfunction, True, save_plot=False)
