## Required software and hardware:

The Python packages required include CV2, Pytorch, Matplotlib and Numpy.
### Installs needed:
- python (3.9 +) 
- OpenCV:
Installed with: `pip install opencv-python`
- Matplotlib:
Installed with: `pip install matplotlib`
- PyTorch:
Installed with: `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
Or if you don't have access to an Nvidia GPU: `pip3 install torch torchvision torchaudio`


## Steps to run:

Record a video of CS:GO gameplay involving some kills from the player. Note that the video must be 1080p and run at 60fps. 

Run the following line without the <>

`python predict.py <path to the video you want to analyze>`


While running, the program will display a window that shows the gameplay that is currently being analyzed along with the number of clips recorded so far. An additional window shows approximately what the parser is looking at. The program will run until either the video is finished, or the user presses ‘q’. Once finished, the clips will automatically be saved to a new folder.

The clips are automatically sent to the CNN for extraction. Once the features have been extracted, the program automatically predicts whether the clips involved the use of the aimbot and the output can be seen in the terminal. 

## Git repo:

https://github.com/SrikarValluri/aimbot-detection

## Unrealized features:
