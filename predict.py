import os
import sys
import random
import subprocess


# print(len(sys.argv))

assert len(sys.argv) >= 2, "Requires path to input file"
inFile = sys.argv[1]
assert os.path.isfile(inFile), f'{inFile} isn\'t a valid file path'

dir = "temp"
delDir = True

if len(sys.argv) > 2:
    dir = sys.argv[2]
    assert os.path.isdir(dir), f'{dir} isn\'t a valid dirctory'
    assert len(os.listdir(dir)) == 0, "flder must be empty"
    delDir = False

if delDir:
    while os.path.isdir(dir):
        dir = "temp" + str(random.randint(1,1000000))
    os.mkdir(dir)

print(f'python auto_clip.py \'{inFile}\' \'{dir}\' 1')
subprocess.run(['python', './auto_clip.py', inFile, dir, '1'])

print('\n\n\n\n\nNow extracting features')

subprocess.run(['python', 'save_cnn_output.py', dir, dir])


print('\n\n\n\n\nNow Predicting value of cheats')


subprocess.run(['python', 'test_rnn.py', os.path.join(dir, 'clips.pt'), dir])

