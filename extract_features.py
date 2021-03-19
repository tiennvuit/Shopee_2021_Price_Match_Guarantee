import json
from sklearn.manifold import TSNE
from PIL import Image
import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

from time import time
model = EfficientNet.from_pretrained('efficientnet-b0')
path = 'train_images/029b2053e294c26f4a86a871bcddda9c.jpg'
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

img = tfms(Image.open(path)).unsqueeze(0)

start = time()
features = model.extract_features(img).flatten().detach().numpy().reshape(-1,1)

X_embedded = TSNE(n_components=2048, method = 'exact').fit_transform(features)

print(features.shape)
print(time() - start)