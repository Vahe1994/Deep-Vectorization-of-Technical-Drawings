from glob import glob
import sys

from PIL import Image

sys.path.append('/code')
from util_files.data.graphics.graphics import VectorImage
from util_files.rendering.cairo import render

if __name__ == '__main__':
    input_dir = '/data/svg_datasets/whole_images/precision-floorplan/test'
    output_dir = '/data/test_sets/pfp'

    for filename in glob(f'{input_dir}/*.svg'):
        sample = filename[len(input_dir)+1:-4]
        print(f'Processing {sample}')
        vector_image = VectorImage.from_svg(filename)
        vector_image.remove_filled()
        vector_image.leave_width_percentile(5)
        vector_image.scale_to_width('min', 1)
        vector_image.save(f'{output_dir}/{sample}.svg')
        Image.fromarray(vector_image.render(render)).save(f'{output_dir}/{sample}.png')
