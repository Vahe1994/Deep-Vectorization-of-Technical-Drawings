#!/usr/bin/env bash
export src_dir='/data/datasets/abc.unpacked/0005/step'
export dst_dir='/data/datasets/svg_datasets/whole_images/abc/with_artefacts'

process_step_file(){
	src="$1"
	samplename="$(basename "${src%.*}")"
	dst="${dst_dir}/${samplename}.svg"
	dirname="$(dirname "$dst")"
    mkdir -p "$dirname"
	echo "Processing $samplename"
	python /trinity/home/o.voinov/work/3ddl/vectorization/abc_line_renders/render_step_file.py \
	  -i "$src" -o "$dst" --width 1024 --height 1024 --line-width 1
}
export -f process_step_file

find $src_dir -type f -name '*.step' | parallel process_step_file
