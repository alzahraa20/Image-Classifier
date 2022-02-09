import argparse 
import time
import torch 
import numpy as np
import json
import sys

from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image

def load_model():
    model_info = torch.load(args.model_checkpoint)
    model = model_info['model']
    model.classifier = model_info['classifier']
    model.load_state_dict(model_info['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    # TODO: Process a PIL image for use in a PyTorch model
    transformed_im = transform(pil_image)
    numpy_im = np.array(transformed_im)
    
    return numpy_im

def classify_image(image_path, topk=5):
    topk=int(topk)
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()
        model = load_model()
        if (args.gpu):
           image = image.cuda()
           model = model.cuda()
        else:
            image = image.cpu()
            model = model.cpu()
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(topk)
        probs, classes = probs[0].tolist(), classes[0].add(1).tolist()
        results = zip(probs,classes)
        return results

def read_categories():
    if (args.category_names is not None):
        cat_file = args.category_names 
        jfile = json.loads(open(cat_file).read())
        return jfile
    return None
        
def display_prediction(results):
    cat_file = read_categories()
    i = 0
    for p, c in results:
        i = i + 1
        p = str(round(p,4) * 100.) + '%'
        if (cat_file):
            c = cat_file.get(str(c),'None')
        else:
            c = ' class {}'.format(str(c))
        print("{}.{} ({})".format(i, c,p))
    return None
    
def parse():
    parser = argparse.ArgumentParser(description='use a neural network to classify an image!')
    parser.add_argument('image_input', help='image file to classifiy (required)')
    parser.add_argument('model_checkpoint', help='model used for classification (required)')
    parser.add_argument('--top_k', help='how many prediction categories to show [default 5].')
    parser.add_argument('--category_names', help='file for category names')
    parser.add_argument('--gpu', action='store_true', help='gpu option')
    args = parser.parse_args()
    return args

def main():
    global args
    args = parse() 
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")
    if (args.top_k is None):
        top_k = 5
    else:
        top_k = args.top_k
    image_path = args.image_input
    prediction = classify_image(image_path,top_k)
    display_prediction(prediction)
    return prediction

main()