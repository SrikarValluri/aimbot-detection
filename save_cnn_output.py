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
import re

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

allVideos = []

subfolders = [ f for f in os.scandir(framesDir) if f.is_dir() ]

with torch.no_grad():
    for folder in subfolders:
        print(f'Etracting Features from clip at {(folder.name).replace("_", ":")}\t\t\t\n', end='', flush=True)

        frames = [ int(re.findall(r'\d+', f.name)[0]) for f in os.scandir(folder.path) if f.is_file() ]

        assert len(frames) == 60, f'{folder.name} has an incorrect number of frames ({len(frames)})'

        frames.sort()

        video = []

        for frame in frames:
            image = cv2.imread(os.path.join(folder.path, f'frame{frame}.png'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = transforms.ToTensor()
            tensor = transform(image)
            tensor = tensor.unsqueeze(0)  
            frameFeatures = model(tensor)
            video.append(frameFeatures.squeeze())

        video = torch.stack(video)
        allVideos.append(video)

print("\nSaving Features")
allVideos = torch.stack(allVideos)
print(allVideos.shape)
torch.save(allVideos, os.path.join(outputDir, "clips.pt"))
print("Features Saved!")
