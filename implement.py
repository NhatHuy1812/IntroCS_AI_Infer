import torch
from torchvision import transforms
from PIL import Image, ImageOps
from models import get_model
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.cnn import TrafficSignModel
import numpy as np
import os
import warnings

def pad_to_square(image):
    width, height = image.size

    if height > width:
        # Cut the bottom to match the width
        cropped_image = image.crop((0, 0, width, width))  # Crop height to width
    elif width > height:
        # Cut equally from both sides to match the height
        delta = (width - height) // 2
        cropped_image = image.crop((delta, 0, width - delta, height))  # Crop width to height
    else:
        return image  # Already square

    return cropped_image

def load_image(image_path, img_cols, img_rows, bbox):
    x1, y1, x2, y2 = bbox
    transform = transforms.Compose([
        transforms.Resize((img_rows, img_cols)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = image.crop((x1, y1, x2, y2)) 
    image = pad_to_square(image)
    image.save('outputs/crop.jpg')
    image = transform(image).unsqueeze(0)
    return image

def load_model(extractor, classifier, localizer_path):
    
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    net1 = get_model('sillnet', 43, 43, 6)
    net1.to(device='cuda')
    net1 = torch.load(extractor,map_location=torch.device('cuda'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net2 = TrafficSignModel().to(device)
    net2.load_state_dict(torch.load(classifier, map_location=device))
    net2.eval()

    stn_weight = torch.hub.load('ultralytics/yolov5', 'custom', path=localizer_path, force_reload=True)
    stn_weight.conf = 0.35

    return (net1, net2, stn_weight)

def localizer(image_path, stn_weight):
    original_image = Image.open(image_path).convert('RGB')
    width, height = original_image.size
    results = stn_weight(original_image)
    detections = results.xyxy[0]
    best_conf = 0
    x1, y1, x2, y2 = (0, 0, height, width)
    for *box, conf, cls_idx in detections:
        if conf > best_conf:
            best_conf = conf
            x1, y1, x2, y2 = list(map(float, box))

    return (x1, y1, x2, y2)

def inference(net, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net1, net2, stn_weight = net

    bbox = localizer(image_path, stn_weight)

    image = load_image(image_path, 64, 64, bbox).to(device)
    with torch.no_grad():
        feat_sem, feat_illu, _ = net1.extract(image, is_warping=True)
        
        recon_feat_sem = net1.decode(feat_sem)
        recon_feat_illu = net1.decode(feat_illu)

    os.makedirs('outputs', exist_ok=True)
    torchvision.utils.save_image(recon_feat_sem, '{}/sem_feature.jpg'.format('outputs'), nrow=8, padding=2)
    torchvision.utils.save_image(recon_feat_illu, '{}/illu_feature.jpg'.format('outputs'), nrow=8, padding=2)

    image = Image.open('outputs/sem_feature.jpg')
    image = image.resize((30,30))
    image = np.array(image)
    image_tensor = torch.tensor(image.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = net2(image_tensor)
        _, pred_class = torch.max(outputs, 1)

    return pred_class.item()

    
