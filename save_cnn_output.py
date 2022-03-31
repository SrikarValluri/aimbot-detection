import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 5
batch_size = 4
learning_rate = 0.001

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()
model.fc = nn.Identity()

rootdir = "./hacks_data_nn/"
i = 0
j = 0
all_videos = []
for subdir, dirs, files in os.walk(rootdir):
    single_video = []
    for file in files:
        # print(os.path.join(subdir, file))
        image = cv2.imread(os.path.join(subdir, file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.ToTensor()
        tensor = transform(image)
        tensor = tensor.unsqueeze(0)  
        frame_features = model(tensor)
        # print(frame_features.shape)
        single_video.append(frame_features.squeeze())
        i += 1

    if(len(single_video) == 50):
        single_video = torch.stack(single_video)
        all_videos.append(single_video)
        torch.save(torch.stack(all_videos), './hacks_data_tensor/hacks_data_tensor_file.pt')
        j += 1

    print(j)
    if(j == 45):
        break

all_videos = torch.stack(all_videos)
print(all_videos.shape)

