from utils import dataset_helpers, prediction_helpers, label_helpers
import numpy as np

def run():

    dataset_path = dataset_helpers.root_path() / 'timeseries_dataset'
    dataset_path.mkdir(exist_ok=True)

    for aoi_id in dataset_helpers.get_aoi_ids('spacenet7'):

            prob_timeseries = prediction_helpers.load_prediction_timeseries('spacenet7', aoi_id,
                                                                      dataset_helpers.include_masked())
            prob_timeseries_file = dataset_path / 'prob_timeseries' / f'prob_timeseries_{aoi_id}.npy'
            prob_timeseries_file.parent.mkdir(exist_ok=True)
            np.save(str(prob_timeseries_file), prob_timeseries.astype(np.float16))

            change_label = label_helpers.generate_change_label('spacenet7', aoi_id, dataset_helpers.include_masked())
            change_label_file = dataset_path / 'change_label' / f'change_{aoi_id}.npy'
            change_label_file.parent.mkdir(exist_ok=True)
            np.save(str(change_label_file), change_label.astype(np.uint8))

            # TODO: add features and change date label



if __name__ == '__main__':
    run()
    pass