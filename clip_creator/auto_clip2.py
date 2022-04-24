import cv2
import numpy as np
import sys

assert len(sys.argv) == 2, "needs file input"

print(sys.argv[1])

# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture(sys.argv[1])

fc = 0

nlines = 0
plines = 0


decCount = 0

pois = []


def countLines(im):
    assert im.shape[1] == 3

    streak = 0
    breakS = 0
    lineC = 0

    for x in range(im.shape[0]):
        if breakS >= 2:
            breakS = 0
            if streak > 15:
                lineC += 1
            streak = 0


        if abs(int(im[x,1,2]) - int(im[x,0,2])) < 20:
            breakS += 1
            continue


        if im[x,1,2] > 150 and im[x,1,1] < 10 and im[x,1,0] < 50:
            streak += 1
            breakS = 0

        else:
            breakS += 1


    
    if breakS > 15:
        lineC += 1

    return lineC




# Loop until the end of the video
while (cap.isOpened()):

	# Capture frame-by-frame
    ret, frame = cap.read()

    if frame is None:
        break

    # frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
	# 					interpolation = cv2.INTER_CUBIC)

    view = frame[70:300, 1800:1919]
    # area of video to look at
    frame = frame[70:300, 1907:1910]



    # print(nlines)

    nlines = countLines(frame)

    print(f'{nlines}\r', end='', flush=True)


    decCount += 1

    if nlines < plines:
        decCount = 0

    if nlines > plines and decCount > 10:
        print("Increase")
        pois.append(fc)

    plines = nlines


	# Display the resulting frame
    cv2.imshow('Frame', frame)

    cv2.imshow('View', view)



	# define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


    fc += 1




# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()

print(pois)
print(len(pois))
