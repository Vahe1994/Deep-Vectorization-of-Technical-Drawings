#!/usr/bin/env python3

import argparse
import os
from itertools import product

import numpy as np
import skimage.io as skio
import torch

from util_files.patchify import patchify


def clean_image(rgb, cleaning_model):
    """Run cleaning operation on the input line drawing,
    leaving clean (non-dirty, non-shaded) and
    full (without gaps) line drawing.

    :param rgb: raw input RGB image
    :type rgb: numpy.ndarray
    :param cleaning_model: an instance of the cleaning model
    :type cleaning_model: callable
    :returns cleaned_rgb -- a cleaner RGB image of the line drawing.
    :rtype cleaned_rgb: numpy.ndarray
    """
    # ensure the channel dimension is present and there's 3 channels
    if len(rgb.shape) < 3:
        rgb = np.repeat(rgb[None], 3, axis=0)

    # ensure the channel dimension is the first one
    if np.argmin(rgb.shape) != 0:
        # assuming it's (h,w,c)
        rgb = rgb.transpose([2,0,1])

    # ensure the input is [0,1] data
    rgb = (rgb / rgb.max()).astype(np.float32)

    # cleaning model wants the image dimensions to be divisible by 8
    h, w = rgb.shape[1:]
    pad_h = ((h - 1) // 8 + 1) * 8 - h
    pad_w = ((w - 1) // 8 + 1) * 8 - w
    input_np = np.pad(rgb, [(0, 0), (0, pad_h), (0, pad_w)], mode='constant', constant_values=1)

    input = torch.from_numpy(np.ascontiguousarray(input_np[None])).cuda()
    cleaned, _ = cleaning_model(input)
    return cleaned[0, 0].cpu().detach().numpy()[..., None]


def split_to_patches(rgb, patch_size, overlap=0):
    """Separates the input into square patches of specified size.

    :param rgb: input RGB image
    :type rgb: numpy.ndarray
    :param patch_size: size of patches in pixels (assuming
                        square patches)
    :type patch_size: int
    :param overlap: amount in pixels of how much the patches
                    can overlap with each other (useful for merging)
    :type overlap: int

    :returns patches, patches_offsets
    :rtype Tuple[numpy.ndarray, numpy.ndarray]
    """
    # TODO @artonson: add correct handling of rightmost patches (currently ignored)
    height, width, channels = rgb.shape
    assert patch_size > 0 and 0 <= overlap < patch_size
    patches = patchify(rgb,
                       patch_size=(patch_size, patch_size, channels),
                       step=patch_size - overlap)
    patches = patches.reshape((-1, patch_size, patch_size))
    height_offsets = np.arange(0, height, step=patch_size - overlap)
    width_offsets = np.arange(0, width, step=patch_size - overlap)
    patches_offsets = np.array(list(
        product(height_offsets, width_offsets)
    ))
    return patches, patches_offsets


def save_output(output_vector, output_filename):
    """

    :param output_vector:
    :param output_filename:
    :return:
    """
    with open(output_filename, 'w') as output_file:
        for primitive in output_vector:
            primitive.write(output_file)


def vectorize(rgb, vector_model):
    pass


def assemble_vector_patches(patches_vector, patches_offsets):
    for patch_vector, patch_offset in zip(patches_vector, patches_offsets):
        for primitive in patch_vector:
            primitive.offset(patch_offset)

    pass


def main(options):
    input_rgb = skio.imread(options.input_filename)

    cleaned_rgb = input_rgb
    if options.use_cleaning:
        cleaning_model = torch.load(options.cleaning_model_filename)
        cleaning_model.eval()
        cleaned_rgb = clean_image(cleaned_rgb, cleaning_model)
        if options.cleaned_filename:
            skio.imsave(options.cleaned_filename, cleaned_rgb)

    if options.vectorize:
        vector_model =  load_vector_model(options.vector_model_filename)
        if options.use_patches:
            patches_rgb, patches_offsets = split_to_patches(cleaned_rgb, options.patch_size)
            #patches_vector = []
            #for patch_idx, patch_rgb in enumerate(patches_rgb):
            #    patch_vector = vectorize(patch_rgb, vector_model)
            #    if options.vector_patch_path:
            #        patch_output_filename = \
            #            os.path.join(options.vector_patch_path,
            #                         'patch_{0:02d}.svg'.format(patch_idx))
            #        save_output(patch_vector, patch_output_filename)

            #    patches_vector.append(patch_vector)
            #output_vector = assemble_vector_patches(patches_vector, patches_offsets)
        else:
            output_vector = vectorize(cleaned_rgb, vector_model)
    
    if options.output_filename:
        save_output(output_vector, options.output_filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0].')
    parser.add_argument('-np', '--no-patches', action='store_false', dest='use_patches', default=True,
                        help='Set to disable vectorization via patches [default: use patches].')
    parser.add_argument('-s', '--patch-size', type=int, dest='patch_size', default=64,
                        help='Patch size in pixels [default: 64].')

    parser.add_argument('-nc', '--no-cleaning', action='store_false', dest='use_cleaning', default=True,
                        help='Set to disable cleaning [default: use cleaning].')
    parser.add_argument('-c', '--cleaning-model-file', dest='cleaning_model_filename',
                        help='Path to cleaning model file [default: none].')

    parser.add_argument('-nv', '--no-vectorization', action='store_false', dest='vectorize', default=True,
                        help='Set to disable vectorization [default: vectorize].')
    parser.add_argument('-v', '--vector-model-file', dest='vector_model_filename',
                        help='Path to vectorization model file [default: none].')

    parser.add_argument('-i', '--input-file', required=True, dest='input_filename',
                        help='Path to input image file [default: none].')
    parser.add_argument('-oc', '--cleaned-file', dest='cleaned_filename',
                        help='Path to cleaned image file [default: none, meaning don\'t save the file].')
    parser.add_argument('-op', '--patch-output-path', dest='patch_output_filename',
                        help='Path to directory containing vectorized patches [default: none, meaning don\'t save the files].')
    parser.add_argument('-o', '--output-file', required=False, dest='output_filename',
                        help='Path to input vector SVG file [default: none].')
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    main(options)
