# aimbot-detection

Step 1:

Reduce 5 second clips in the "hacks" and "no_hacks" folder to 1 second cropped clips in "hacks_data" and "no_hacks_data" by calling extract_1_sec.py

Step 2:

Convert each video that is stored within "hacks_data" and "no_hacks_data" into 50 snapshots each in their own folders, and store each folder in "hacks_data_nn" and "no_hacks_data_nn"

Step 3:

Convert each sequence of pictures/frames into a sequence vector that has a pre-trained resnet CNN applied to it. Store in "hacks_data_tensor" and "no_hacks_data_tensor"

Step 4:

Create and train a recurrent neural network (LSTM) using the tensor data (all manipulation/organization of data done here) and train for aimbot or non-aimbot behavoir detection. Accuracy of the model is then calculated.

## Required software and hardware:

The Python packages required include CV2, Pytorch, Matplotlib and Numpy.
###### Installs needed:
