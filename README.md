## Required Software Packages:
- python (3.9 +) 
- OpenCV:
Installed with: `pip install opencv-python`
- Matplotlib:
Installed with: `pip install matplotlib`
- PyTorch:
Installed with: `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
Or if you don't have access to an Nvidia GPU: `pip3 install torch torchvision torchaudio`


## Finding and predicting clips in a video:

Record a video of CS:GO gameplay involving some kills from the player. 
**Note that the video must be 1080p and run at 60fps.**

Predict by running: `python predict.py <video>` where `<video>` contains the path to the video you want to make predictions on

While running, the program will display a window that shows the gameplay that is currently being analyzed along with the number of clips recorded so far. An additional window shows approximately what the parser is looking at. The program will run until either the video is finished, or the user presses ‘q’. Once finished, the clips will automatically be saved to a new folder.

The clips are automatically sent to the CNN for extraction. Once the features have been extracted, the program automatically predicts whether the clips involved the use of the aimbot and the output can be seen in the terminal. 

## Training the model
1. Record gameplay of CS:GO at 1080p 60fps
2. Parse gameplay using `autoclip.py` 
	- aimbot clips should be saved to `hacks_data_nn`
	- regular clips should be saved to `no_hacks_data_nn`
3. Extract clip features using `save_cnn_output.py`
	- aimbot clips should be saved to `hacks_data_tensor`
	- regular clips should be saved to `no_hacks_data_tensor`
4. Train RNN using `train_rnn.py`
5. Model will be saved to `models/model.pt` **Any file with that name will be overwritten**


## Using files outside of pipeline
### Parser `autoclip.py`
To run the auto clipper by itself, run the following command: `python ./auto_clip.py <input file> <output directory> <use time>`
- `<input file>` is the file you want to parse
- `<output directory>` is where you want the clips to be output to
- `<use time>` (Optional, defaults to `0`)  adds a timestamp to the clips. Set to `1` to turn the timestamps on, and `0` to turn them off

### CNN `save_cnn_output.py`
To run the CNN by itself, run the following command: `python save_cnn_output.py <clip directory> <output directory>`
- `<clip directory>` is the directory where the clips are located
- `<output directory>` is where you want to save the extracted features 

### Test RNN `test_rnn.py`
To run the extracted features through the RNN, run the following command: `python test_rnn.py <.pt path> <clip path>`
- `<.pt path>` is the path to the .pt file output from the CNN
- `<clip path>` (Optional, displays timestamps when specified) is the path to clips corresponding to the .pt file are saved

### Train RNN `train_rnn.py`
To train the RNN, run the following command: `python train_rnn.py`

### Validate RNN `validate_rnn.py`
To validate the RNN, run the following command: `python validate_rnn.py`

## Git repo:

https://github.com/SrikarValluri/aimbot-detection

## Unrealized features:
- Cheat detection for multiple games and different cheats
- Website for easy access and use
