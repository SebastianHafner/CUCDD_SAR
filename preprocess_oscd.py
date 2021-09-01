from utils import geofiles, dataset_helpers
import numpy as np
from pathlib import Path


def produce_oscd_change_labels(path_oscd_dataset: Path, path_oscd_multitemporal_dataset: Path):
    root_path = path_oscd_multitemporal_dataset
    aoi_ids = [f.stem for f in root_path.iterdir() if f.is_dir()]
    for aoi_id in aoi_ids:
        change_label_file = path_oscd_dataset / 'labels' / aoi_id / 'cm' / f'{aoi_id}-cm.tif'
        # does not contain geographical information
        change, _, _ = geofiles.read_tif(change_label_file)

        # change label value range from [1, 2] to [0, 1] by subtracting one
        change = change - 1

        # reading geographical information from sentinel 2 file
        s2_path = path_oscd_multitemporal_dataset / aoi_id / 'sentinel2'
        s2_file = [f for f in s2_path.glob('**/*') if f.is_file()][0]
        _, geotransform, crs = geofiles.read_tif(s2_file)

        to_file = path_oscd_multitemporal_dataset / aoi_id / f'change_{aoi_id}.tif'
        geofiles.write_tif(to_file, change.astype(np.uint8), geotransform, crs)
    pass


def prepare_oscd_change_labels():
    # getting all aoi ids
    oscd_labels_path = dataset_helpers.oscd_path() / 'labels'
    aoi_ids = [d.stem for d in oscd_labels_path.iterdir() if d.is_dir()]
    for aoi_id in aoi_ids:
        cm_file = dataset_helpers.oscd_path() / 'labels' / aoi_id / 'cm' / f'{aoi_id}-cm.tif'
        # no geographical information
        cm, _, _ = geofiles.read_tif(cm_file)
        cm = cm - 1
        s2_folder = dataset_helpers.oscd_path() / 'images' / aoi_id / 'imgs_1'
        s2_file = [f for f in s2_folder.glob('**/*') if f.is_file() and f.stem.split('_')[-1] == 'B02'][0]
        _, geotransform, crs = geofiles.read_tif(s2_file)

        output_file = dataset_helpers.oscd_path() / 'labels_gee' / f'change_{aoi_id}.tif'
        geofiles.write_tif(output_file, cm, geotransform, crs)


def get_date(file: Path) -> tuple:
    parts = file.stem.split('_')
    return int(parts[-2]), int(parts[-1])


# creates dict with aoi_ids as keys and with timestamps of the multitemporal oscd dataset as values
def assemble_oscd_timestamps():
    data = {}
    aoi_paths = [f for f in dataset_helpers.dataset_path('oscd').iterdir() if f.is_dir()]

    # processing each aoi
    for aoi_path in aoi_paths:
        # initializing aoi
        aoi_id = aoi_path.name

        # get s1 timestamps
        s1_path = dataset_helpers.dataset_path('oscd') / aoi_id / 'sentinel1'
        s1_dates = [get_date(f) for f in s1_path.glob('**/*') if f.is_file()]

        # get s2 timestamps
        s2_path = dataset_helpers.dataset_path('oscd') / aoi_id / 'sentinel2'
        s2_dates = [get_date(f) for f in s2_path.glob('**/*') if f.is_file()]

        all_dates_sorted = sorted(list(set(s1_dates + s2_dates)), key=lambda d: d[0] * 12 + d[1])
        timestamps = [[year, month, False] for year, month in all_dates_sorted]

        data[aoi_id] = timestamps

    output_file = dataset_helpers.dataset_path('oscd') / 'oscd_timestamps.json'
    geofiles.write_json(output_file, data)


def generate_oscd_metadata_file():
    timestamps = dataset_helpers.oscd_timestamps()
    bad_data = dataset_helpers.bad_data('ocsd')

    data = {
        's1_bands': ['VV', 'VH'],
        's2_bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B6', 'B8', 'B8A', 'B11', 'B12'],
        'yx_sizes': {},
        'split': {'train': [], 'test': []},
        'aois': {}
    }

    # add training and test sites to metadata
    train_file = dataset_helpers.oscd_path() / 'images' / 'train.txt'
    with open(str(train_file)) as f:
        content = f.read()[:-1].split(',')
        data['split']['train'] = content
    test_file = dataset_helpers.oscd_path() / 'images' / 'test.txt'
    with open(str(test_file)) as f:
        content = f.read()[:-1].split(',')
        data['split']['test'] = content

    for aoi_id in timestamps.keys():

        # put together metadata for aoi_id
        aoi_data = []
        aoi_timestamps = timestamps[aoi_id]
        yx_size_set = False
        for i, timestamp in enumerate(aoi_timestamps):
            year, month, mask = timestamp

            # check if satellite data is ok for timestamp in aoi based on bad data file
            s1 = False if i in bad_data[aoi_id]['S1'] else True
            s2 = False if i in bad_data[aoi_id]['S2'] else True
            timestamp_data = [year, month, mask, s1, s2]
            aoi_data.append(timestamp_data)

            if not yx_size_set and s1:
                s1_path = dataset_helpers.dataset_path('oscd') / aoi_id / 'sentinel1'
                s1_file = s1_path / f'sentinel1_{aoi_id}_{year}_{month:02d}.tif'
                arr, _, _ = geofiles.read_tif(s1_file)
                data['yx_sizes'][aoi_id] = (arr.shape[0], arr.shape[1])
                yx_size_set = True

        data['aois'][aoi_id] = aoi_data

    output_file = dataset_helpers.dataset_path('oscd') / f'metadata.json'
    geofiles.write_json(output_file, data)


if __name__ == '__main__':
    # assemble_oscd_timestamps()
    generate_oscd_metadata_file()
    # prepare_oscd_change_labels()
