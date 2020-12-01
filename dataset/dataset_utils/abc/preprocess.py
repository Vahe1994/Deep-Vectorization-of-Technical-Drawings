import sys

import torch
from tqdm import tqdm

sys.path.append('/code')
from util_files.data.preprocessed import _PreprocessedDataset, svg_parameters, patch_parameters
from util_files.data.vectordata.preprocessed import PreprocessedSVGPacked


def preprocess_beziers_only():
    # %% Parameters
    num_workers = 15
    batch_size = 256

    shuffle_seed = 67
    patch_parameters['max_lines'] = 0
    patch_parameters['max_curves'] = 10

    svg_parameters['source_pattern'] = '**/*'

    for subset in 'val', 'train':
        data_root = f'/data/svg_datasets/patched/abc/everything/{subset}'
        out_root = f'/data/svg_datasets/preprocessed/abc/everything_is_quad_bezier/{subset}'

        # %% Prepare data
        dataset = _PreprocessedDataset.from_svgdataset(
            patch_parameters=patch_parameters, svg_parameters=svg_parameters, out_root=out_root, shuffle_seed=shuffle_seed,
            preprocessed_svg_dataset_class=PreprocessedSVGPacked,
            data_root=data_root)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # %% Preprocess
        for _ in tqdm(dataloader):
            pass
