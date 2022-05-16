import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


# Defining LSTM RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()     # inheriting from existing RNN class
        self.num_layers = num_layers    # number of input layers
        self.hidden_size = hidden_size  # number of hidden players

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # creating LSTM layer
        self.fc = nn.Linear(hidden_size, num_classes)                               # creating linear output layer

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


        return torch.sigmoid(out) # returning one forward step of the NN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_classes = 1
num_epochs = 100

learning_rate = 0.001

input_size = 512
sequence_length = 50
hidden_size = 512
num_layers = 2

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device) # Creating instance of LSTM model

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Loading the model
hacks_data = torch.load("./hacks_data_tensor/full_data/hacks_data_tensor_file_large.pt")
hacks_labels = torch.ones(hacks_data.shape[0]).unsqueeze(1)

no_hacks_data = torch.load("./no_hacks_data_tensor/full_data/no_hacks_data_tensor_file_large.pt")
no_hacks_labels = torch.zeros(no_hacks_data.shape[0]).unsqueeze(1)


# Seperating Training/Testing data into 90%/10% splits
hacks_data_train = hacks_data[:int(len(hacks_data) * 0.9)]
hacks_data_test = hacks_data[int(len(hacks_data) * 0.9):]

no_hacks_data_train = no_hacks_data[:int(len(no_hacks_data) * 0.9)]
no_hacks_data_test = no_hacks_data[int(len(no_hacks_data) * 0.9):]

hacks_labels_train = hacks_labels[:int(len(hacks_labels) * 0.9)]
hacks_labels_test = hacks_labels[int(len(hacks_labels) * 0.9):]

no_hacks_labels_train = no_hacks_labels[:int(len(no_hacks_labels) * 0.9)]
no_hacks_labels_test = no_hacks_labels[int(len(no_hacks_labels) * 0.9):]


train_data = torch.cat((hacks_data_train, no_hacks_data_train))
train_labels = torch.cat((hacks_labels_train, no_hacks_labels_train))


test_data = torch.cat((hacks_data_test, no_hacks_data_test))
test_labels = torch.cat((hacks_labels_test, no_hacks_labels_test))

model.train() # set model to training mode 
for epoch in range(num_epochs):
    images = train_data         # using our set of images
    labels = train_labels       # using our set of labels
    labels = labels.to(device)  # uploading onto CPU/GPU

    random_shuffle = torch.randperm(images.size()[0])  # shuffling the data for every epoch
    images = images[random_shuffle]
    labels = labels[random_shuffle]
    
    # Forward pass
    outputs = model(images)     # perform a forward pass 
    loss = criterion(outputs, labels) # calculate loss/error
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') # print results for every epoch

    torch.save(model, "./models/model.pt") # incrementally save model
