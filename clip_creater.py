import cv2
import numpy as np
#import plotly.express as px

def decrement_island(img, x, y):
    size = 0
    if x < 0 or x >= len(img) or y < 0 or y >= len(img[0]) or img[x][y] != 255:
        return size
    img[x][y] = 254
    size += 1
    size += decrement_island(img, x-1, y-1)
    size += decrement_island(img, x-1, y)
    size += decrement_island(img, x-1, y+1)
    size += decrement_island(img, x, y-1)
    size += decrement_island(img, x, y+1)
    size += decrement_island(img, x+1, y-1)
    size += decrement_island(img, x+1, y)
    size += decrement_island(img, x+1, y+1)
    return size



cap = cv2.VideoCapture('testKnife.mkv')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

fc = 0
ret = True
ret2 = True
ret3 = True

#while (fc < (frameCount-2)  and ret):
 #   ret, buf[fc] = cap.read()
  #  fc += 1

#img = cv2.cvtColor(buf[45], cv2.COLOR_BGR2GRAY)
#crop_test = img[1060:frameHeight, 1890:frameWidth]
#ret, thresh_test = cv2.threshold(crop_test, 180, 255, cv2.THRESH_BINARY)

#pix_count = 0
#for i in thresh_test:
#    for j in i:
#        if j == 255:
#            pix_count += 1
#    print(i)
#print(pix_count)



no_bull = []
move_bull = []
has_num = []
add_num = False
fired = []

for i in range(0, frameCount-2):
    total_island = 0
    total_pix = 0
    ret, temp_img = cap.read()
    temp_gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
    ret2, temp_thresh = cv2.threshold(temp_gray, 180, 255, cv2.THRESH_BINARY)
    ret3, temp_thresh2 = cv2.threshold(temp_gray, 240, 255, cv2.THRESH_BINARY)
    temp_crop = temp_thresh2[1030:frameHeight, 1700:frameWidth]
    temp_crop2 = temp_thresh[1060:frameHeight, 1890:frameWidth]
    #print(buf[i, 1056, 1877])
    if temp_gray[1056, 1877] < 180:
        #if buf[i, 1056, 1877][1] < 180:
         #   if buf[i, 1056, 1877][2] < 180:
        #print(str(i) + " = no bullet")
        no_bull.append(i)
    #for j in range(1030, frameHeight):
     #   for k in range(1700, frameWidth):
      #      if temp_thresh2[j, k] != 0:
       #         add_num = True
        #        break
   # if add_num:
    #    has_num.append(i)
     #   add_num = False
    for j in range(0, len(temp_crop2)):
        for k in range(0, len(temp_crop2[j])):
            if temp_crop2[j][k] == 255:
                total_pix = decrement_island(temp_crop2, j, k)
                total_island += 1
    if(total_island == 1):
        if(26>total_pix>10):
            move_bull.append(i)
    if i == 5135:
    #    detector = cv2.SimpleBlobDetector_create()
     #   keypoints = detector.detect(temp_crop2)
      #  im_with_keypoints = cv2.drawKeypoints(temp_crop2, keypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
       # cv2.imshow("Keypoints", im_with_keypoints)
        #cv2.waitKey(0)
    #    nblobs = len(keypoints)
     #   print(nblobs)
        print(total_island)
        print(total_pix)
        cv2.namedWindow('frame 10')
        cv2.imshow('frame 10', temp_img)
        cv2.waitKey(0)
        cv2.namedWindow('frame 10')
        cv2.imshow('frame 10', temp_crop2)
        cv2.waitKey(0)


print(no_bull)
#print(move_bull)
#print(has_num)
for i in range(0, len(no_bull)):
    for j in range(0, len(move_bull)):
        if no_bull[i] == move_bull[j]:
            #for k in range(0, len(has_num)):
             #   if no_bull[i] == has_num[k]:
            fired.append(no_bull[i])
            break
print(fired)

#img_rgb = img##np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ##    [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
              ## ], dtype=np.uint8)
#fig = px.imshow(img_rgb)
#fig.show()


cap.release()
