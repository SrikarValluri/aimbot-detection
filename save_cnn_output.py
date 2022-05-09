import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import sys

if(len(sys.argv) != 3):
    framesDir = "./hacks_data_nn"
    outputDir = "./hacks_data_tensor/no_hacks_data_tensor_file"
    

else:
    framesDir = sys.argv[1]
    outputDir = sys.argv[2]

if(not os.path.exists(framesDir)):
    print("Frames directory is invalid.")
    sys.exit()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()
model.fc = nn.Identity()

i = 0
j = 0
dirCt = 0
allVideos = []

# rootdir = "./AA_On_nn"
# savedName = './no_hacks_data_tensor/no_hacks_data_tensor_file'
# savedName = './AA_On_tensor/AA_On_tensor_file'
for subdir, dirs, files in os.walk(framesDir):
    singleVideo = []
    for file in files:
        image = cv2.imread(os.path.join(subdir, file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.ToTensor()
        tensor = transform(image)
        tensor = tensor.unsqueeze(0)  
        frameFeatures = model(tensor)
        singleVideo.append(frameFeatures.squeeze())
        i += 1

    if(len(singleVideo) == 60):
        singleVideo = torch.stack(singleVideo)
        allVideos.append(singleVideo)
        torch.save(torch.stack(allVideos), (outputDir + str(dirCt) + ".pt"))
        j += 1

    print(j)
    if(j % 5 == 0 and j != 0):
        allVideos.clear()
        dirCt += 5
        print("saving_new")

allVideos = torch.stack(allVideos)
print(allVideos.shape)

