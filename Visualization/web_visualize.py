import os
import glob2
import flask
from flask import Flask
from flask import render_template
import pandas as pd
import numpy as np

IMAGE_FOLDER = os.path.join('./static', 'train_images')

app = Flask(__name__)
app.config['IMAGE_FODLER'] = IMAGE_FOLDER






@app.route('/', methods=['GET'])
def visualize():

    # Load image files and ground truth files
    image_names = sorted(glob2.glob(os.path.join(app.config['IMAGE_FODLER'], '*.jpg')))[:1000]

    groundtruth = pd.read_csv('static/train.csv')[:1000]

    data = np.array([image_names, groundtruth['title'].to_list()])

    data = [
        data[:, 0:len(image_names):4].T,
        data[:, 1:len(image_names):4].T,
        data[:, 2:len(image_names):4].T,
        data[:, 3:len(image_names):4].T
    ]

    # Load ground truths and get informatio
    return render_template('base.html', data=data, image_names=image_names)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    