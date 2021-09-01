from pathlib import Path
from utils import geofiles, dataset_helpers
from tqdm import tqdm
import numpy as np


# helper function to get date from label file name
# global_monthly_2020_01_mosaic_L15-1335E-1166N_5342_3524_13_Buildings.geojson
def get_date(label_file: Path) -> tuple:
    file_name = label_file.stem
    file_parts = file_name.split('_')
    year, month = int(file_parts[2]), int(file_parts[3])
    return year, month


# helper function to check if mask exists for time stamp
def has_mask(aoi_id: str, year: int, month: int) -> bool:
    mask_path = dataset_helpers.spacenet7_path() / 'train' / aoi_id / 'UDM_masks'
    mask_file = mask_path / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_UDM.tif'
    return mask_file.exists()


# creates dict with aoi_ids as keys and with timestamps of SpaceNet7 dataset as values
def assemble_spacenet7_timestamps():
    dataset_path = dataset_helpers.spacenet7_path() / 'train'
    data = {}
    aoi_paths = [f for f in dataset_path.iterdir() if f.is_dir()]

    # processing each aoi
    for aoi_path in aoi_paths:

        # initializing aoi
        aoi_id = aoi_path.name

        # getting timestamps from labels
        labels_path = aoi_path / 'labels_match'
        label_files = [f for f in labels_path.glob('**/*')]
        dates = [get_date(label_file) for label_file in label_files]

        # sort timestamps by dates (ascending)
        dates = sorted(dates, key=lambda date: date[0] * 12 + date[1])
        aoi_data = [(*d, has_mask(aoi_id, *d)) for d in dates]
        data[aoi_id] = aoi_data

    output_file = dataset_helpers.dataset_path('spacenet7') / 'spacenet7_timestamps.json'
    geofiles.write_json(output_file, data)


# merges buildings from a time series for a site into one geojson file
def assemble_spacenet7_buildings():
    dataset_path = dataset_helpers.spacenet7_path() / 'train'
    aoi_paths = sorted([f for f in dataset_path.iterdir() if f.is_dir()])

    for aoi_path in aoi_paths:
        aoi_id = aoi_path.name
        print(f'processing aoi: {aoi_id}')
        all_buildings = None

        labels_path = aoi_path / 'labels_match'
        label_files = sorted([f for f in labels_path.glob('**/*')])

        for label_file in tqdm(label_files):
            data = geofiles.load_json(label_file)
            if all_buildings is None:
                all_buildings = dict(data)
                all_buildings['features'] = []

            year, month = get_date(label_file)
            features = []
            for f in data['features']:
                p = f['properties']
                properties = {'Id': p['Id'], 'year': year, 'month': month}
                f['properties'] = properties
                features.append(f)

            all_buildings['features'].extend(features)

        output_file = dataset_helpers.spacenet7_path() / 'buildings_assembled' / f'buildings_{aoi_id}.geojson'
        geofiles.write_json(output_file, all_buildings)


# merges masks from a time series into a single geotiff file
def assemble_spacenet7_masks():

    aoi_ids = dataset_helpers.get_aoi_ids('spacenet7', exclude_missing=False)

    for aoi_id in aoi_ids:
        print(f'processing aoi: {aoi_id}')

        masks_path = dataset_helpers.spacenet7_path() / 'train' / aoi_id / 'UDM_masks'
        mask_files = [f for f in masks_path.glob('**/*') if f.is_file()]

        if not mask_files:
            continue

        def date_value(file: Path):
            _, _, year, month, *_ = file.stem.split('_')
            return int(year) * 12 + int(month)
        mask_files = sorted(mask_files, key=lambda f: date_value(f))

        masks = None
        for i, mask_file in enumerate(tqdm(mask_files)):

            mask, geotransform, crs = geofiles.read_tif(mask_file, first_band_only=True)
            if masks is None:
                masks = np.zeros((*mask.shape, len(mask_files)), dtype=np.uint8)

            masks[:, :, i] = mask / 255

        output_file = dataset_helpers.spacenet7_path() / 'masks_assembled' / f'masks_{aoi_id}.tif'
        geofiles.write_tif(output_file, masks, geotransform, crs)


def generate_spacenet7_metadata_file():

    timestamps = dataset_helpers.spacenet7_timestamps()
    bad_data = dataset_helpers.bad_data('spacenet7')
    missing_aois = dataset_helpers.missing_aois()

    data = {
        's1_bands': ['VV', 'VH'],
        's2_bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B6', 'B8', 'B8A', 'B11', 'B12'],
        'yx_sizes': {},
        'aois': {}
    }

    for aoi_id in timestamps.keys():

        # skip aoi if it's missing
        if aoi_id in missing_aois:
            continue

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
                s1_path = dataset_helpers.dataset_path('spacenet7') / aoi_id / 'sentinel1'
                s1_file = s1_path / f'sentinel1_{aoi_id}_{year}_{month:02d}.tif'
                arr, _, _ = geofiles.read_tif(s1_file)
                data['yx_sizes'][aoi_id] = (arr.shape[0], arr.shape[1])
                yx_size_set = True

        data['aois'][aoi_id] = aoi_data

    output_file = dataset_helpers.dataset_path('spacenet7') / f'metadata.json'
    geofiles.write_json(output_file, data)


if __name__ == '__main__':
    # assemble_spacenet7_timestamps()
    generate_spacenet7_metadata_file()
    # assemble_spacenet7_masks()
    # assemble_spacenet7_buildings()
    # generate_spacenet7_dataset_file(ROOT_PATH / 'continuous_urban_change_detection' / 'spacenet7_s1s2_dataset')