from pathlib import Path
from utils import geofiles, config
import preprocess_spacenet7


def bad_data() -> dict:
    bad_data_file = Path.cwd() / 'bad_data' / f'bad_data_{config.dataset_name()}.json'
    bad_data = geofiles.load_json(bad_data_file)
    return bad_data


def missing_aois() -> list:
    file = Path.cwd() / 'missing_aois.json'
    missing = geofiles.load_json(file)
    return missing


def timestamps() -> dict:
    timestamps_file = config.dataset_path() / 'spacenet7_timestamps.json'
    if not timestamps_file.exists():
        preprocess_spacenet7.assemble_spacenet7_timestamps()
    assert(timestamps_file.exists())
    timestamps = geofiles.load_json(timestamps_file)
    return timestamps


def metadata() -> dict:
    metadata_file = config.dataset_path() / 'cucdd_sar_metadata.json'
    if not metadata_file.exists():
        preprocess_spacenet7.generate_spacenet7_metadata_file()
    assert (metadata_file.exists())
    metadata = geofiles.load_json(metadata_file)
    return metadata


def aoi_metadata(aoi_id: str) -> list:
    md = metadata()
    return md['aois'][aoi_id]


def metadata_index(aoi_id: str, year: int, month: int) -> int:
    md = metadata()[aoi_id]
    for i, (y, m, *_) in enumerate(md):
        if y == year and month == month:
            return i


def metadata_timestamp(aoi_id: str, year: int, month: int) -> int:
    md = metadata()[aoi_id]
    for i, ts in enumerate(md):
        y, m, *_ = ts
        if y == year and month == month:
            return ts


def date2index(date: list) -> int:
    ref_value = 2019 * 12 + 1
    year, month = date
    return year * 12 + month - ref_value


# include masked data is only
def get_timeseries(aoi_id: str) -> list:
    aoi_md = aoi_metadata(aoi_id)
    timeseries = [[y, m, mask, s1] for y, m, mask, s1 in aoi_md if s1]

    # trim time series at beginning and end such that it starts and ends with an unmasked timestamp
    unmasked_indices = [i for i, (_, _, mask, _) in enumerate(timeseries) if not mask]
    min_unmasked, max_unmasked = min(unmasked_indices), max(unmasked_indices)
    timeseries = timeseries[min_unmasked:max_unmasked + 1]
    return timeseries


def length_timeseries(aoi_id: str) -> int:
    ts = get_timeseries(aoi_id)
    return len(ts)


def duration_timeseries(aoi_id: str, include_masked_data: bool = False) -> int:
    start_year, start_month = get_date_from_index(0, aoi_id, include_masked_data)
    end_year, end_month = get_date_from_index(-1, aoi_id, include_masked_data)
    d_year = end_year - start_year
    d_month = end_month - start_month
    return d_year * 12 + d_month


def get_date_from_index(index: int, aoi_id: str, include_masked_data: bool = False) -> tuple:
    ts = get_timeseries(aoi_id, include_masked_data)
    year, month, *_ = ts[index]
    return year, month


def get_aoi_ids(exclude_missing: bool = True, min_timeseries_length: int = None) -> list:
    if config.subset_activated():
        aoi_ids = config.subset_aois()
    else:
        ts = timestamps()
        aoi_ids = [aoi_id for aoi_id in ts.keys() if not (exclude_missing and aoi_id in missing_aois())]
    if min_timeseries_length is not None:
        all_aoi_ids = aoi_ids
        aoi_ids = []
        for aoi_id in all_aoi_ids:
            ts_length = length_timeseries(aoi_id)
            if ts_length >= min_timeseries_length:
                aoi_ids.append(aoi_id)
    return sorted(aoi_ids)


def get_geo(aoi_id: str) -> tuple:
    folder = config.dataset_path() / aoi_id / 'sentinel1'
    file = [f for f in folder.glob('**/*') if f.is_file()][0]
    _, transform, crs = geofiles.read_tif(file)
    return transform, crs


def get_yx_size(aoi_id: str) -> tuple:
    md = metadata()
    return md['yx_sizes'][aoi_id]


def date2str(date: list) -> str:
    year, month, *_ = date
    return f'{year-2000:02d}-{month:02d}'



