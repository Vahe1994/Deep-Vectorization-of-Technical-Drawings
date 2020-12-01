import os
import sys

sys.path.append('/code')
from util_files.data.graphics.graphics import VectorImage


def remove_duplicates_and_trim(src_filename, dst_filename):
    vector_image = VectorImage.from_svg(src_filename)
    vector_image.remove_duplicates()
    vector_image.remove_filled()
    vector_image.adjust_viewbox()
    vector_image.adjust_view()
    os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
    vector_image.save(dst_filename)


if __name__ == '__main__':
    remove_duplicates_and_trim(sys.argv[1], sys.argv[2])
