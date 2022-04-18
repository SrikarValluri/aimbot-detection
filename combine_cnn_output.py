import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


directory = "./no_hacks_data_tensor/"

new_tensors = []
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".pt"):
        new_tensors.append(torch.load(directory+str(filename)))
    

full_tensor = torch.cat(new_tensors)
print(full_tensor.shape)
torch.save(full_tensor, directory + "full_data/no_hacks_data_tensor_file.pt")