from tqdm import tqdm


def quantitative_testing_dataset(model: cd_models.ChangeDetectionMethod):
    preds, gts = [], []
    for aoi_id in tqdm(dataset_helpers.get_aoi_ids()):
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

    print(f'F1: {f1:.3f} - P: {precision:.3f} - R: {recall:.3f}')


def ablation1(error_multiplier: float, min_diff_range: tuple, step_size: float, band: str = 'VV'):
    min_diff_candidates = np.range(*min_diff_range, step_size)
    for min_diff_candidate in tqdm(min_diff_candidates):
        sf = cd_models.StepFunctionModel(band, error_multiplier=error_multiplier, min_diff=min_diff_candidate,
                                         min_segment_length=2)


def ablation2(min_diff: float, error_multiplier_range: tuple, step_size: float, band: str = 'VV'):
    pass


if __name__ == '__main__':
    pass

