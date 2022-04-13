import argparse
import collections
import datetime
import pathlib
import cv2
import numpy as np
from keyclipwriter import KeyClipWriter
import argparse
import datetime
import time
import cv2

cap = cv2.VideoCapture("cheatarms1.mp4")
kills = 0
frames = 0
l = collections.deque(maxlen=550)

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
    help="path to output directory")
ap.add_argument("-p", "--picamera", type=int, default=-1,
    help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-f", "--fps", type=int, default=60,
    help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
    help="codec of output video")
ap.add_argument("-b", "--buffer-size", type=int, default=550,
    help="buffer size of video clip writer")

kcw = KeyClipWriter(bufSize=550)
consecFrames = 0

result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'xvid'),
                         60, (1080, 1920))
while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    updateConsecFrames = False
    
    lower_red = np.array([30,200,50])
    upper_red = np.array([255,255,180])

    
    sky = frame[70:110, 1600:1920]
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    sky2 = mask[70:110, 1600:1920]

    other = cv2.bitwise_and(sky, sky, mask= sky2)

    cv2.imshow('Video', sky)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.imshow('other', other)
    l.append(frame)
    tot_pixel = sky.size
    red_pixel = np.count_nonzero(sky2)
    percentage = round(red_pixel * 100 / tot_pixel, 2)
    if(percentage > 2 and percentage < 4):
        if(frames == 0):
            updateConsecFrames = True
            kills = kills + 1
            frames = frames + 1
            print("Kills : %i", kills)
            print("Kill detected\n")
            if not kcw.recording:
                timestamp = datetime.datetime.now()
                p = "{}/{}.avi".format(pathlib.Path(__file__).parent.resolve(),
                    timestamp.strftime("%Y%m%d-%H%M%S"))
                kcw.start(p, cv2.VideoWriter_fourcc(*"MJPG"),
                    60)

        else:
            updateConsecFrames = True

            frames = frames + 1
            if kcw.recording:
                kcw.finish()
            if(frames > 550):
                frames = 0
    if updateConsecFrames:
        consecFrames += 1
    # update the key frame clip buffer
    kcw.update(frame)
    # if we are recording and reached a threshold on consecutive
    # number of frames with no action, stop recording the clip
    if kcw.recording and consecFrames == 100:
        kcw.finish()
        
        
    if(percentage < 1):
        frames = 0

    

    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()