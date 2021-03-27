import os
import glob2


def load_img_from_dir(path):
    img_paths = sorted(glob2.glob(os.path.join(path, '*.jpg')))[:10]
    return img_paths


def load_features_from_dir(path):
    feature_files = sorted(glob2.glob(os.path.join(path, '*.pt')))[:10]
    return feature_files


def load_query_imgs_from_file(file_path):
    with open(file_path, 'r') as f:
        query_imgs = sorted([line.rstrip() for line in f.readlines()])[:10]
    return query_imgs
