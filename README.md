# Urban change detection and dating using Sentinel time series data: Exploring the potential of a robust urban extractor

# To do:
- 2021-05-06 Rework bad data handling. Single sensor data should be used in case data from either sensor is corrupted.
(tried this one but doesn't work!)
- 2021-05-06 Add masks to SpaceNet7_S1S2_Dataset and ad support for masks
- 2021-05-06 Fix OSCD Multitemporal Dataset Labels (size does not correspond to satellite images)
- 2021-05-15 Fix edge effects in change detection and dating label
- 2021-05-15 Add sizes of aois to metadata file
- 2021-05-15 Add bad data to time series length plot
- 2021-05-15 Write evaluation script for model confidence (i.e., compute metrics for confidence groups)
- 2021-05-15 Handle missing AOIs
- 2021-05-15 Cut-off values for urban extraction results?
- 2021-05-15 Add deep features for SpaceNet7 and OSCD (as numpy arrays to save storage)

# Added changes:
- 2021-05-14 Reworked data preprocessing
- 2021-05-14 Added support for spacenet7_s1s2_dataset_v2 and switching between datasets is not possible
(but broke OSCD dataset)
- 2021-05-14 Added time series length plot