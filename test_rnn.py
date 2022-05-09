import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

assert len(sys.argv) >= 2, "File requires input path"

inFile = sys.argv[1]
assert os.path.isfile(inFile), "not a valid file"

inDir = False

if len(sys.argv) > 2:
    inDir = sys.argv[2]
    assert os.path.isdir(inDir), "not a valid directory"



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # x -> (batch_size, seq_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # out -> (batch_size, seq_size, input_size) = (N, 50, 512)
        out = out[:, -1, :]
        # out -> (N, 512)
        out = self.fc(out)


        return torch.sigmoid(out)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load("./models/model_max_data.pt")

data = torch.load(inFile)

times = []

if inDir:
    times = [ (f.name).replace("_", ":") for f in os.scandir(inDir) if f.is_dir() ]
    assert len(times) == data.shape[0], "number of clips doesnt match feature data"


model.eval()
with torch.no_grad():

    for i in range(len(data)):
        curr_test = data[i].unsqueeze(0)
        output = model(curr_test)
        output = output.item()

        print(f'Clip at {times[i]}: {"Regular Gameplay Detected" if output < 0.5 else "Aimbot Detected"}')
