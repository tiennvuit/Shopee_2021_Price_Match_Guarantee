import os
import glob2
import argparse
from tqdm import tqdm


def create_img_files_from_dir(path):

    if not os.path.exists(path):
        print("Invalid path !")
        exit(0)

    img_lst = sorted(glob2.glob(os.path.join(path, '*.jpg')))

    with open('image_files.txt', 'w') as f:
        for img_file in tqdm(img_lst):
            f.write(img_file+'\n')
    print("Saved result to {} file".format('image_files.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create image file list for extract features.')
    parser.add_argument('--input_dir', required=True,
        help='The folder directory. contain image files.'
    )

    args = vars(parser.parse_args())

    create_img_files_from_dir(path=args['input_dir'])