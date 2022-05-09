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


if(len(sys.argv) != 2):
    inputDir = "./hacks_data_tensor/"
    # directory = "./AA_On_tensor/"
else:
    inputDir = sys.argv[1]

if(not os.path.exists(inputDir)):
    print("Input directory is invalid.")

# (from bradley's code) inure outDir ends with /
inputDir = inputDir + ("/" if inputDir[-1] != "/" else "")

newTensors = []
for file in os.listdir(inputDir):
     filename = os.fsdecode(file)
     if filename.endswith(".pt"):
        newTensors.append(torch.load(inputDir+str(filename)))
    

fullTensor = torch.cat(newTensors)
print(fullTensor.shape)
torch.save(fullTensor, inputDir + "full_data/" + inputDir[:-1] + "_file.pt")
