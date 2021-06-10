#!/usr/bin/env bash
export src_dir='/data/svg_datasets/whole_images/abc/with_artefacts'
export dst_dir='/data/svg_datasets/whole_images/abc/train'

postprocess_svg(){
	src="$1"
	filename="$(basename "$src")"
	dst="${dst_dir}/$filename"
	echo "Processing $filename"
	python /code/scripts/dataset_utils/abc/remove_duplicates_and_trim.py "$src" "$dst"
}
export -f postprocess_svg

find $src_dir -type f -name '*.svg' | parallel postprocess_svg
