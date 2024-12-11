import torch
from torchvision import transforms
from PIL import Image
from models import get_model
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.cnn import TrafficSignModel
import numpy as np
import os
import warnings


def load_image(image_path, img_cols, img_rows):
    transform = transforms.Compose([
        transforms.Resize((img_rows, img_cols)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def load_model(extractor, classifier):
    
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    net1 = get_model('sillnet', 43, 43, 6)
    net1.to(device='cuda')
    net1 = torch.load(extractor,map_location=torch.device('cuda'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net2 = TrafficSignModel().to(device)
    net2.load_state_dict(torch.load(classifier, map_location=device))
    net2.eval()

    return (net1, net2)


def inference(net, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net1, net2 = net
    image = load_image(image_path, 64, 64).to(device)
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

    
