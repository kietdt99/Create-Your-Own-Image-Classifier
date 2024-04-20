import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import torchvision.models as models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type = str, default='flowers/test/1/image_06752.jpg')
    parser.add_argument('--load_dir', default = "checkpoint.pth", type = str)
    parser.add_argument('--top_k', default = 5, type = int)
    parser.add_argument('--category_names', type = str)
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()

def load_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    model = torchvision.models.vgg16(pretrained=True)
    
    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    transformers = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    tensor_img = transformers(image)

    return tensor_img


def predict(image_path, model, topk, device):
    model.to(device)

    img = Image.open(image_path)
    img_torch = process_image(img)
    
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    img_torch = img_torch.to(device)

    with torch.no_grad():
        output = model.forward(img_torch)

    probability = F.softmax(output.data, dim=1)

    return probability.topk(topk)

def load_cat_names(category_names):
    if category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
    return cat_to_name

def formatCompletedPercentage(raw_precent):
    percent_split = raw_precent.split(".")
    percent_first = percent_split[0]
    percent_last = percent_split[1]
    if len(percent_first) == 1:
        percent_first = "0" + percent_first
    if len(percent_last) == 1:
        percent_last = percent_last + "0"
    return percent_first + "." + percent_last + "%"

def main():
    args = parse_args()
    device = 'cuda' if args.gpu else 'cpu'
    print(f'Using {device} device')    
    model = load_checkpoint(args.load_dir)
    cat_names = load_cat_names(args.category_names)
    number_classes = args.top_k
    image_path= args.image_dir
    to_parse = predict(image_path, model, number_classes, device)
    probabilities = to_parse[0][0].cpu().numpy()
    mapping = {val: key for key, val in model.class_to_idx.items()}

    classes = to_parse[1][0].cpu().numpy()
    classes = [mapping [item] for item in classes]
    classes = [cat_names[str(index)] for index in classes]

    for l in range(number_classes):
        raw_percent = f"{probabilities [l]*100:.2f}"
        print(f"Probability {formatCompletedPercentage(raw_percent)} is {classes[l]}.")

if __name__ == "__main__":
    main()