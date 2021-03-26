"""
Usage:


CUDA_VISIBLE_DEVICES=7 python extract_features.py --image_file image_file.txt --pre_model vgg16 --output_dir data/extracted_features/

"""

import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from resnet_pytorch import ResNet 
from torchvision import transforms
import torchvision.models as models


PREDEFINED_MODELS = {
    'vgg16': models.vgg16(pretrained=True),
    'vgg19': models.vgg19(pretrained=True),
    'alex': models.alexnet(pretrained=True),
    'resnet101': ResNet.from_pretrained('resnet101')
}




def main(args):
    
    if not os.path.exists(args['image_file']):
        print("INVALID image file.")
        exit(0)

    print("[info] Finding availabel images")
    img_paths = []
    with open(args['image_file'], 'r') as f:
        image_files = f.readlines()
        for image_path in tqdm(image_files):
            if not os.path.exists(image_path.rstrip()):
                print("Not found {}".format(image_path))
            img_paths.append(image_path.rstrip())
    print("--> Found {} images".format(len(img_paths)))

    # Load model
    print("[info] Loading {} pretrained model".format(args['pre_model']))
    model = PREDEFINED_MODELS[args['pre_model']]
    if args['pre_model'] in ['alex', 'vgg16', 'vgg19']:
        model.classifier = model.classifier[:-1]
    model.eval()

    # Preprocess data
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    if not os.path.exists(os.path.join(args['output_dir'], args['pre_model'])):
        os.mkdir(os.path.join(args['output_dir'], args['pre_model']))

    for i, img_path in enumerate(tqdm(img_paths)):
        ouptut_file = os.path.join(args['output_dir'], args['pre_model'], os.path.basename(img_path).split(".")[0]+'.pt')
        if os.path.exists(ouptut_file):
            continue
        img = tfms(Image.open(img_path)).unsqueeze(0)
        if args['pre_model'] in ['alex', 'vgg16', 'vgg19']:
            extracted_features = model(img)
        else:
            extracted_features = model.extract_features(img)
        torch.save(extracted_features, ouptut_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features of list of images.')
    parser.add_argument('--image_file', required=True,
        help='The file container list of images need extract features.'
    )
    parser.add_argument('--pre_model', required=True, choices=['vgg16', 'vgg19', 'alex', 'resnet101'],
        help='The model using extract features.'
    )
    parser.add_argument('--output_dir', required=True,
        help='The path store extracted features.'
    )

    args = vars(parser.parse_args())

    main(args)