from utils import dataset_helpers, geofiles
import numpy as np


def mask_index(dataset: str, aoi_id: str, year: int, month: int) -> int:
    md = dataset_helpers.metadata()['aois'][aoi_id]
    md_masked = [[y, m, mask, *_] for y, m, mask, *_ in md if mask]
    for i, (y, m, *_) in enumerate(md_masked):
        if year == y and month == m:
            return i


def has_mask(aoi_id: str, year: int, month: int) -> bool:
    md = dataset_helpers.metadata()['aois'][aoi_id]
    for y, m, mask, *_ in md:
        if year == y and m == month:
            return mask


def has_masked_timestamps(aoi_id: str) -> bool:
    ts = dataset_helpers.get_timeseries(aoi_id, include_masked_data=True)
    ts_masked = [[y, m, mask, *_] for y, m, mask, *_ in ts if mask]
    return True if ts_masked else False


# if no mask exists returns false for all pixels
def load_mask(aoi_id: str, year: int, month: int) -> np.ndarray:
    if has_mask(aoi_id, year, month):
        index = mask_index(aoi_id, year, month)
        masks = load_masks(aoi_id)
        return masks[:, :, index]
    else:
        return np.zeros(shape=dataset_helpers.get_yx_size(aoi_id), dtype=np.bool)


def load_masks(aoi_id: str) -> np.ndarray:
    masks_file = dataset_helpers.dataset_path() / aoi_id / f'masks_{aoi_id}.tif'
    assert(masks_file.exists())
    masks, *_ = geofiles.read_tif(masks_file)
    return masks.astype(np.bool)


def is_fully_masked(aoi_id: str, year: int, month: int) -> bool:
    mask = load_mask(aoi_id, year, month)
    n_elements = np.size(mask)
    n_masked = np.sum(mask)
    # TODO: mismatch due to GEE download probabably
    if n_elements * 0.9 < n_masked:
        return True
    return False


if __name__ == '__main__':
    pass
