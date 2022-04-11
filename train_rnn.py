import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2



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

# Hyper-parameters 
num_classes = 1
num_epochs = 100

learning_rate = 0.001

input_size = 512
sequence_length = 50
hidden_size = 512
num_layers = 2

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Loading the model
hacks_data = torch.load("./hacks_data_tensor/hacks_data_tensor_file.pt")
hacks_labels = torch.ones(hacks_data.shape[0]).unsqueeze(1)

print(hacks_data.shape)
print(hacks_labels.shape)

no_hacks_data = torch.load("./no_hacks_data_tensor/no_hacks_data_tensor_file.pt")
no_hacks_labels = torch.zeros(hacks_data.shape[0]).unsqueeze(1)


# Seperating Training/Testing data
hacks_data_train = hacks_data[:20]
hacks_data_test = hacks_data[20:]

no_hacks_data_train = no_hacks_data[:20]
no_hacks_data_test = no_hacks_data[20:]

hacks_labels_train = hacks_labels[:20]
hacks_labels_test = hacks_labels[20:]

no_hacks_labels_train = no_hacks_labels[:20]
no_hacks_labels_test = no_hacks_labels[20:]


train_data = torch.cat((hacks_data_train, no_hacks_data_train))
train_labels = torch.cat((hacks_labels_train, no_hacks_labels_train))



print(train_data.shape)
print(train_labels.shape)

test_data = torch.cat((hacks_data_test, no_hacks_data_test))
test_labels = torch.cat((hacks_labels_test, no_hacks_labels_test))

model.train()
for epoch in range(num_epochs):
    images = train_data
    labels = train_labels
    labels = labels.to(device)

    random_shuffle = torch.randperm(images.size()[0])
    images = images[random_shuffle]
    labels = labels[random_shuffle]
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if(epoch > 75 and (0.15 < loss.item() < 0.22)):
        break

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for i in range(len(test_data)):
        curr_test = test_data[i].unsqueeze(0)
        curr_label = test_labels[i][0].item()
        outputs = model(curr_test)

        if(abs(outputs.item()-curr_label) < 0.5):
            print("success")
            n_correct += 1

        n_samples += 1

print(n_samples)
print("percentage correct: " + str(n_correct/n_samples * 100.))



torch.save(model, "./models/model.pt")
