#!/usr/bin/env bash

export splits_dir='/gpfs/gpfs0/3ddl/vectorization/datasets/svg_datasets/whole_images/precision-floorplan'
export root_dir='/gpfs/gpfs0/3ddl/vectorization/datasets/svg_datasets/patched/precision-floorplan'

for prim_subset in 'everything' 'curves_only' 'lines_only'; do
    for train_subset in 'val' 'test'; do
        mkdir -p "${root_dir}/${prim_subset}/${train_subset}"
        find "${splits_dir}/${train_subset}" -type f -exec sh -c '
            samplename=$(basename "${0%.*}")
            mv "'${root_dir}/${prim_subset}/train/'${samplename}" "'${root_dir}/${prim_subset}/${train_subset}/'${samplename}"
        ' {} ';'
    done
done
