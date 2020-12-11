## Dataset utils

Data generation for ABC
1. Generate renders from CAD dataset_utils/abc/render_step_file.py --width 1024 --height 1024 --line-width 1 …(already done in downloadable abc dataset)
2. get rid off duplicates in dataset_utils/abc/move_duplicates.sh(already done in downloadable abc dataset)
3. deleting identical primitives and cutting voids /dataset_utils/abc/remove_duplicates_and_trim.py src dst(already done in downloadable abc dataset)
4. Patchifying and filtering junks dataset_utils/abc/make_patches.py src dst_dir,\
    in this case small primitives, simplifaing curves, skiping pathces with overlays, transforming lines to curves
5. Memory maping to speed up dataset loading for training /dataset_utils/abc/preprocess.py 
