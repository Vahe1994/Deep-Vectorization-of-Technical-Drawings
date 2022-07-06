import argparse
import sys
sys.path.append("../../")
from cleaning.utils.synthetic_data_generation import Synthetic
from tqdm import tqdm


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--img_path', type=str, default='../../dataset/Cleaning/Synthetic/train', help='Path to save data')
    parser.add_argument('--data_count', type=int, default=7, help='Generated data count')
    parser.add_argument('--data_count_start', type=int, default=0, help='Generated data count start')
    args = parser.parse_args()
    return args


def main(args):
    syn = Synthetic()
    for it in tqdm(range(args.data_count_start, args.data_count)):
        syn.get_image(img_path=args.img_path, name=str(it))


if __name__ == '__main__':
    args = parse_args()
    main(args)
