import os
import argparse
from utils.similarity_metrics import manhattan, euclid, cosine
from utils.read_write_data import load_img_from_dir, load_features_from_dir, load_query_imgs_from_file
from extract_features import PREDEFINED_MODELS
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import torch


def query(query_imgs, img_paths, feature_paths, args):
    """
        Query image to get the top highest similar images 
    """
    # Load all feature files
    feature_list = np.array([torch.load(file_path).detach().numpy().reshape(-1) for file_path in feature_paths])

    # Load pretrained model
    model = PREDEFINED_MODELS[args['model']]
    if args['model'] in ['alex', 'vgg16', 'vgg19']:
        model.classifier = model.classifier[:-1]
    model.eval()

    # Define the preprocess images
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    print("Starting process query imagess")
    with open(args['output'], 'a+') as out_f:
        for i, query in enumerate(query_imgs):
            print("{}/{} Quering image {}".format(i+1, len(query_imgs), os.path.basename(query)))
            img = tfms(Image.open(query)).unsqueeze(0)
            if args['model'] in ['alex', 'vgg16', 'vgg19']:
                extracted_features = model(img).detach().numpy().reshape(-1, 1)
            else:
                extracted_features = model.extract_features(img).detach().numpy().reshape(1, -1)
        
            # Get the ranking results base on the similarity metric.
            similarities = euclid(extracted_features, feature_list)
            ranking_list = np.argsort(similarities)
            if args['metric'] == 'cosine':
                ranking_list = ranking_list[::-1]

            # write the searching result to output file
            write_line = os.path.basename(query) + ', ' + ' '.join([os.path.basename(img_paths[x]) for x in ranking_list])
            for x in ranking_list:
                out_f.write(write_line + '\n')
    return ranking_list, similarities


def main(args):

    # Load all extracted features with corresponding images
    img_paths = load_img_from_dir(path='./data/train_images/')
    feature_paths = load_features_from_dir(path=os.path.join('./data/extracted_features/', args['model']))
    query_imgs = load_query_imgs_from_file(file_path=args['query_img'])

    print("[info] The number of images in database: {}".format(len(img_paths)))
    print("[info] The number of feature files in database: {}".format(len(feature_paths)))

    # searching
    top_results, similarities = query(query_imgs=query_imgs,
                        img_paths=img_paths,
                        feature_paths=feature_paths,
                        args=args)

    # Save results
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Query the image and return list of results')
    parser.add_argument('--query_img', required=True,
        help='The path of file container query images.'
    )
    parser.add_argument('--model', required=True, 
        choices=['vgg16', 'vgg19', 'alex', 'resnet101'],
        help='The model to extract features from image.'
    )
    parser.add_argument('--metric', required=True, choices=['cosine', 'euclid', 'manhattan'],
        help='The metric evaluate the different between two vectors',
    )
    parser.add_argument('--conf_thres', required=False,
        help='The lower bound of similarity to ranking images'
    )
    parser.add_argument('--number_imgs', required=False,
        help='The number of return results'
    )
    parser.add_argument('--output', required=True,
        help='The output path of query',
    )
    args = vars(parser.parse_args())
    print(args)

    main(args)
