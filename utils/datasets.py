import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

import json

from pathlib import Path
from abc import abstractmethod
from utils import geofiles, dataset_helpers, prediction_helpers, label_helpers, augmentations


class TimeseriesDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, run_type: str):
        super().__init__()
        self.cfg = cfg
        self.run_type = run_type

        self.input = cfg.DATALOADER.INPUT
        self.output = cfg.DATALOADER.OUTPUT

        self.root_path = Path(cfg.DATASETS.ROOT_PATH)
        self.dataset_name = cfg.DATASETS.NAME
        self.data_path = self.root_path / self.dataset_name / f'{self.input}_timeseries'
        self.label_path = self.root_path / self.dataset_name / f'{self.output}_label'

        self.aoi_ids = cfg.DATASETS.AOI_IDS.TRAINING if run_type == 'training' else cfg.DATASETS.AOI_IDS.VALIDATION
        self.samples = []
        self.aoi_seq_lengths = {}
        for aoi_id in self.aoi_ids:
            data = self._load_data_for_aoi(aoi_id)
            m, n, seq_length = data.shape
            if aoi_id not in self.aoi_seq_lengths.keys():
                self.aoi_seq_lengths[aoi_id] = seq_length
            for i in range(m):
                for j in range(n):
                    self.samples.append((aoi_id, i, j))
        self.max_seq_length = max([v for v in self.aoi_seq_lengths.values()])
        self.length = len(self.samples)

        self.transform = augmentations.compose_transformations(cfg)

        self.n_classes = 2

    def __getitem__(self, index: int) -> dict:

        aoi_id, i, j = self.samples[index]

        x = self._load_data(aoi_id, i, j)
        y = self._load_label(aoi_id, i, j)

        x = torch.tensor(x).float()
        y = torch.tensor(y).long()

        # x, y = self.transform((x, y))

        sample = {
            'x': x,
            'y': y,
            'aoi_id': aoi_id,
            'i': i,
            'j': j,
            'length': self._seq_length(aoi_id)
        }

        return sample

    def _load_data_for_aoi(self, aoi_id: str) -> np.ndarray:
        data_file = self.data_path / f'{self.input}_timeseries_{aoi_id}.npy'
        data = np.load(str(data_file)).astype(np.float16)
        return data

    def _load_data(self, aoi_id: str , i: int, j: int) -> np.ndarray:
        data = self._load_data_for_aoi(aoi_id)
        data_padded = np.zeros((self.max_seq_length, 1), dtype=np.float16)
        seq_length = self._seq_length(aoi_id)
        start_index = self.max_seq_length - seq_length
        data_padded[start_index:, 0] = data[i, j, ]
        return data_padded

    def _load_label_for_aoi(self, aoi_id: str) -> np.ndarray:
        label_file = self.label_path / f'{self.output}_{aoi_id}.npy'
        label = np.load(str(label_file)).astype(np.uint8)
        return label

    def _load_label(self, aoi_id: str, i: int, j: int) -> int:
        label = self._load_label_for_aoi(aoi_id)
        # return np.array([label[i, j, ]]).astype(np.uint8)
        return int(label[i, j])

    def _seq_length(self, aoi_id: str) -> int:
        return self.aoi_seq_lengths[aoi_id]

    def __len__(self) -> int:
        return self.length

    def __str__(self) -> str:
        return 'not implemented'

    def class_weights(self) -> list:
        n_class_pixels = np.zeros(self.n_classes)
        bins = np.arange(-0.5, self.n_classes, 1)
        for sample in self.samples:
            aoi_id, i, j = sample
            label = self._load_label(aoi_id, i, j)
            label = np.argmax(label, axis=-1)
            hist_sample, _ = np.histogram(label, bins=bins)
            n_class_pixels += hist_sample
        return n_class_pixels / np.sum(n_class_pixels)

    def sampler(self):
        # TODO: investigate baseline
        pass
        # weights = np.array([float(sample['weight']) for sample in self.samples])
        # sampler = torch_data.WeightedRandomSampler(weights=weights, num_samples=self.length, replacement=True)
        # return sampler
