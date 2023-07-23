from PIL import Image
from torchvision import transforms
from itertools import product
import sys

sys.path.append('/code/Deep-Vectorization-of-Technical-Drawings/')

from vectorization import load_model
from util_files.patchify import patchify
import argparse
import os
import numpy as np
import torch

from refinement.our_refinement.refinement_for_curves import main as curve_refinement

from merging.merging_for_curves import main as curve_merging
from refinement.our_refinement.refinement_for_lines import render_optimization_hard
from merging.merging_for_lines import postprocess



def serialize(checkpoint):
    model_state_dict = checkpoint['model_state_dict']
    keys = []
    for k in model_state_dict:
        if 'hidden.transformer' in k:
            keys.append(k)

    for k in keys:
        new_key = 'hidden.decoder.transformer' + k[len('hidden.transformer'):]
        model_state_dict[new_key] = model_state_dict[k]
        del model_state_dict[k]
    return checkpoint


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
    rgb = rgb.transpose(1, 2, 0)
    rgb_t = np.ones((rgb.shape[0] + 33, rgb.shape[1] + 33, rgb.shape[2])) * 255.
    rgb_t[:rgb.shape[0], :rgb.shape[1], :] = rgb
    rgb = rgb_t

    height, width, channels = rgb.shape

    assert patch_size > 0 and 0 <= overlap < patch_size
    patches = patchify(rgb,
                       patch_size=(patch_size, patch_size, channels),
                       step=patch_size - overlap)
    patches = patches.reshape((-1, patch_size, patch_size, channels))
    height_offsets = np.arange(0, height - patch_size, step=patch_size - overlap)
    width_offsets = np.arange(0, width - patch_size, step=patch_size - overlap)
    patches_offsets = np.array(list(
        product(height_offsets, width_offsets)
    ))
    return patches, patches_offsets, rgb


def preprocess_image(image):
    patch_height, patch_width = image.shape[1:3]
    image = torch.as_tensor(image).type(torch.float32).reshape(-1, patch_height, patch_width) / 255
    image = 1 - image  # 0 -- background
    mask = (image > 0).type(torch.float32)
    _xs = np.arange(1, patch_width + 1, dtype=np.float32)[None].repeat(patch_height, 0) / patch_width
    _ys = np.arange(1, patch_height + 1, dtype=np.float32)[..., None].repeat(patch_width, 1) / patch_height
    _xs = torch.from_numpy(_xs)[None]
    _ys = torch.from_numpy(_ys)[None]
    return torch.stack([image, _xs * mask, _ys * mask], dim=1)


def read_data(options, image_type='RGB'):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = []
    if options.image_name is None:
        image_names = os.listdir(options.data_dir)
        print(image_names)
        for image_name in image_names:
            if (image_name[-4:] != 'jpeg' and image_name[-3:] != 'png' and image_name[-3:] != 'jpg') or image_name[
                0] == '.':
                print(image_name[-4:])
                continue

            img = train_transform(Image.open(options.data_dir + image_name).convert(image_type))
            print(img.shape)
            img_t = torch.ones(img.shape[0], img.shape[1] + (32 - img.shape[1] % 32),
                               img.shape[2] + (32 - img.shape[2] % 32))
            img_t[:, :img.shape[1], :img.shape[2]] = img
            dataset.append(img_t)
        options.image_name = image_names
    else:
        img = train_transform(Image.open(options.data_dir + options.image_name).convert(image_type))
        print(img)
        print(img.shape)
        img_t = torch.ones(img.shape[0], img.shape[1] + (32 - img.shape[1] % 32),
                           img.shape[2] + (32 - img.shape[2] % 32))
        img_t[:, :img.shape[1], :img.shape[2]] = img
        dataset.append(img_t)
        options.image_name = [options.image_name]
    return dataset


def vector_estimation(patches_rgb, model, device, it, options):
    '''
    :param image:
    :param model:
    :param device:
    :param it:
    :param options:
    :return:
    '''
    model.eval()
    patches_vector = []
    print('--- Preprocessing BEGIN')
    patch_images = preprocess_image(patches_rgb)
    print('--- Preprocessing END')
    for it_batches in range(400, patch_images.shape[0] + 399, 400):
        it_start = it_batches - 400
        if it_batches > patch_images.shape[0]:
            it_batches = patch_images.shape[0]
        with torch.no_grad():
            if (it_start == 0):
                patches_vector = model(patch_images[it_start:it_batches].cuda().float(),
                                       options.model_output_count).detach().cpu().numpy()
            else:
                patches_vector = np.concatenate((patches_vector, model(patch_images[it_start:it_batches].cuda().float(),
                                                                       options.model_output_count).detach().cpu().numpy()),
                                                axis=0)
    patches_vector = torch.tensor(patches_vector) * 64
    return patches_vector


def main(options):
    if len(options.gpu) == 0:
        device = torch.device('cpu')
        prefetch_data = False
    elif len(options.gpu) == 1:
        device = torch.device('cuda:{}'.format(options.gpu[0]))
        prefetch_data = True
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(options.gpu)
        device = torch.device('cuda:{}'.format(options.gpu[0]))
        prefetch_data = True
        parallel = True

    ##reading images
    images = read_data(options, image_type='L')



    ##loading model
    model = load_model(options.json_path).to(device)
    checkpoint = serialize(torch.load(options.model_path))
    model.load_state_dict(checkpoint['model_state_dict'])
    ## iterating through images and calculating
    for it, image in enumerate(images):
        image_tensor = image.unsqueeze(0).to(device)
        options.sample_name = options.image_name[it]
        # splitting image
        patches_rgb, patches_offsets, input_rgb = split_to_patches(image_tensor.cpu().numpy()[0] * 255, 64, options.overlap)
        patches_vector = vector_estimation(patches_rgb, model, device, it, options)

        if options.primitive_type == "curve":
            intermediate_output = {'options': options, 'patches_offsets': patches_offsets,
                                   'patches_vector': patches_vector,
                                   'cleaned_image_shape': (image_tensor.shape[1], image_tensor.shape[2]),
                                   'patches_rgb': patches_rgb}
            primitives_after_optimization, patch__optim_offsets, repatch_scale, optim_vector_image = curve_refinement(
                options, intermediate_output, optimization_iters_n=options.diff_render_it)
            merging_result = curve_merging(options, vector_image_from_optimization=optim_vector_image)
        elif options.primitive_type == "line":
            vector_after_opt = render_optimization_hard(patches_rgb, patches_vector, device, options, options.image_name[it])
            merging_result, rendered_merged_image = postprocess(vector_after_opt,patches_offsets,input_rgb,image,0,options)
        else:
            raise ( options.primitive_type+"not implemented, please choose between line or curve")

        return merging_result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', action='append', help='GPU to use, can use multiple [default: use CPU].')
    parser.add_argument('-c', '--curve_count', type=int, default=10, help='curve count in patch [default: 10]')
    parser.add_argument('--primitive_type', type=str, default="line", help='line or curve')
    parser.add_argument('--output_dir', type=str, default="/logs/outputs/vectorization/lines/", help='dir to folder for output')
    parser.add_argument('--diff_render_it', type=int, default=400, help='iteration count')
    parser.add_argument('--init_random', action='store_true', default=False, dest='init_random',
                        help='init model with random [default: False].')
    parser.add_argument('--rendering_type', type=str, default='hard', help='hard -oleg,simple Alexey')
    parser.add_argument('--data_dir', type=str, default="/data/abc_png_svg/", help='dir to folder for input')
    parser.add_argument('--image_name', type=str, default=None,
                        help='Name of image.If None will perform to all images in '
                             'folder.[default: None]')
    parser.add_argument('--overlap', type=int, default=0, help='overlap in pixel')
    parser.add_argument('--model_output_count', type=int, default=10, help='max_model_output')
    parser.add_argument('--max_angle_to_connect', type=int, default=10, help='max_angle_to_connect in pixel')
    parser.add_argument('--max_distance_to_connect', type=int, default=3, help='max_distance_to_connect in pixel')
    parser.add_argument('--model_path', type=str,
                        default="/logs/models/vectorization/lines/model_lines.weights",
                        help='parth to trained model')
    parser.add_argument('--json_path', type=str,
                        default="/code/Deep-Vectorization-of-Technical-Drawings/vectorization/models/specs/resnet18_blocks3_bn_256__c2h__trans_heads4_feat256_blocks4_ffmaps512__h2o__out512.json",
                        help='dir to folder for json file for transformer')
    options = parser.parse_args()

    return options


if __name__ == '__main__':
    options = parse_args()
    main(options)
