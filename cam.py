import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms

import cv2

def generate_cam(feature_conv, weight_softmax, class_idx):
    """Generate a CAM
       
       Paper and methodology link : http://cnnlocalization.csail.mit.edu/

    Arguments:
        feature_conv {[type]} -- [description]
        weight_softmax {[type]} -- [description]
        class_idx {[type]} -- [description]
    """
    upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, upsample))
    
    return output_cam

def cam(network, features, pil, classes, img):
    """Preprocess image and generate CAM

    Paper and methodology link : http://cnnlocalization.csail.mit.edu/
    
    Arguments:
        network {[type]} -- [description]
        features {[type]} -- [description]
        pil {[type]} -- [description]
        classes {[type]} -- [description]
        img {[type]} -- [description]
    """
    
    parameters = list(net.parameters())
    weight_softmax = np.squeeze(parameters[-2].data.cpu().numpy())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225] 
    )
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = network(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    proabilities, idx = h_x.sort(0, True)

    for i in range(0, 2):
        print(f"{proabilities[i]} -> {classes[idx[i].item()]}")
    
    CAMs = 