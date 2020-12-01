#!/usr/bin/env bash

export splits_dir='/data/svg_datasets/whole_images/abc'
export root_dir='/data/svg_datasets/patched/abc'

for prim_subset in 'everything'; do
    for train_subset in 'duplicates' 'val' 'test'; do
        mkdir -p "${root_dir}/${prim_subset}/${train_subset}"
        find "${splits_dir}/${train_subset}" -type f -exec sh -c '
            samplename=$(basename "${0%.*}")
            mv "'${root_dir}/${prim_subset}/train/'${samplename}" "'${root_dir}/${prim_subset}/${train_subset}/'${samplename}"
        ' {} ';'
    done
done
