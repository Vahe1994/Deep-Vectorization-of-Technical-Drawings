#!/usr/bin/env bash
export src_dir='/data/svg_datasets/whole_images/abc/train'
export dst_dir='/data/svg_datasets/patched/abc/everything/train'

make_patches(){
	src="$1"
	filename="$(basename "$src")"
	echo "Processing $filename"
	python /code/scripts/dataset_utils/abc/make_patches.py "$src" "$dst_dir"
}
export -f make_patches

find $src_dir -type f -name '*.svg' | /trinity/home/o.voinov/bin/parallel make_patches
