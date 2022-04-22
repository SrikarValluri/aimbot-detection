import argparse
import collections
import datetime
import math
import pathlib
import cv2
import numpy as np
from keyclipwriter import KeyClipWriter
import argparse
import datetime
import time
import cv2

#Uses code from:
#https://pyimagesearch.com/2016/02/29/saving-key-event-video-clips-with-opencv/
#

cap = cv2.VideoCapture("no_cheats3.mkv")
kills = 0
frames = 0
prev_kills = 0
curr_kills = 0
nored = 1
prev_nored = 1
yes_red = 0
flux = 0

l = collections.deque(maxlen=550)

kcw = KeyClipWriter(bufSize=70)
consecFrames = 0

result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'xvid'),
                         60, (1080, 1920))
while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    updateConsecFrames = False
    
    lower_red = np.array([100,200,100])
    upper_red = np.array([255,255,180])

    
    sky = frame[70:500, 1907:1909]
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    sky2 = mask[70:500, 1907:1909]

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

    curr_kills = round(percentage / 1.22)
    print(percentage)
    if(curr_kills > prev_kills):
        if(yes_red > 10):
            prev_kills = curr_kills
            yes_red = 0
        #record
            if not kcw.recording:
                print("Kills : ", curr_kills)
                timestamp = datetime.datetime.now()
            
                p = "{}/{}.avi".format(pathlib.Path(__file__).parent.resolve(),
                timestamp.strftime("%Y%m%d-%H%M%S"))
                kcw.start(p, cv2.VideoWriter_fourcc(*"MJPG"),
                    60)
        else:
            yes_red = yes_red + 1   

        
        if(nored == 60):
            prev_nored = 0
            nored = 0
        else:
            prev_nored = nored
            nored = nored + 1

    elif(curr_kills < prev_kills):
        if(flux > 10):
            print("Kills : ", curr_kills)
            prev_kills = curr_kills
            flux = 0
        else:
            flux = flux + 1
       

    if kcw.recording:
        kcw.finish()



        


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