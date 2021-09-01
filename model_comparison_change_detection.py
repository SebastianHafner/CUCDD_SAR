import numpy as np
import matplotlib.pyplot as plt
from utils import geofiles, dataset_helpers, label_helpers, prediction_helpers, metrics, visualization
import change_detection_models as cd_models


def qualitative_comparison_change_detection(models: list, dataset: str, aoi_id: str, names: list = None,
                                            save_plot: bool = False):

    fontsize = 16
    dates = dataset_helpers.get_timeseries(ds, aoi_id, dataset_helpers.include_masked())

    n_models = len(models)
    fig, axs = plt.subplots(1, 3 + n_models, figsize=((3 + n_models) * 5, 5))

    # pre image, post image and gt
    visualization.plot_optical(axs[0], dataset, aoi_id, *dates[0][:2])
    axs[0].set_title('S2 Start TS', fontsize=fontsize)
    visualization.plot_optical(axs[1], dataset, aoi_id, *dates[-1][:2])
    axs[1].set_title('S2 End TS', fontsize=fontsize)
    visualization.plot_change_label(axs[2], dataset, aoi_id, dataset_helpers.include_masked())
    axs[2].set_title('Change GT', fontsize=fontsize)

    for i, model in enumerate(models):
        change = model.change_detection(dataset, aoi_id, dataset_helpers.include_masked())
        visualization.plot_blackwhite(axs[3 + i], change)
        title = names[i] if names is not None else model.name
        axs[3 + i].set_title(title, fontsize=fontsize)

    if not save_plot:
        plt.show()
    else:
        dataset_name = dataset_helpers.dataset_name(dataset)
        save_path = dataset_helpers.root_path() / 'plots' / 'comparison_detection' / dataset_name
        save_path.mkdir(exist_ok=True)
        output_file = save_path / f'{aoi_id}_detection.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':

    ds = 'oscd'
    pcc = cd_models.PostClassificationComparison()
    # sf2 = cd_models.StepFunctionModel(n_stable=2)
    sf6 = cd_models.StepFunctionModel(n_stable=6)

    models = [pcc, sf6]
    names = ['PCC', 'SF_6']

    for aoi_id in dataset_helpers.get_aoi_ids(ds):
        qualitative_comparison_change_detection(models, ds, aoi_id, names=names, save_plot=True)
