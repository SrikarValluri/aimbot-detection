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

lscore = 0


# Loop until the end of the video
while (cap.isOpened()):

	# Capture frame-by-frame
    ret, frame = cap.read()

    if frame is None:
        break

    fc += 1

    # frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
	# 					interpolation = cv2.INTER_CUBIC)

    frame = frame[70:300, 1908:1910]

    # frame[10,0] = [0,255,0]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # print(frame[0,0])
    # print("-------")
    # print(hsv[0,0])
    # print("=======")

    s = hsv[:,:,1]
    h = hsv[:,:,0]

    s[s[:] < 200] = 0

    # filter to red hue
    h[h[:] > 174 ] = 255
    h[h[:] < 5] = 255
    h[h[:] < 255] = 0

    r = frame.copy()[:,:,2]

    r[r[:] < 150] = 0

    test = h.copy()

    test[s[:] == 0] = 0
    test[r[:] < 50] = 0

    linesP = cv2.HoughLinesP(
            test, # Input edge image
            3, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=15, # Min number of votes for valid line
            minLineLength=15, # Min allowed length of line
            maxLineGap=1 # Max allowed gap between line for joining them
            )

    if linesP is not None:
        nlines = len(linesP)
    else:
        nlines = 0

    # print(nlines)

    if nlines > plines:
        print("Increase")

    lscore = .75 * lscore + .25 * nlines
    
    print(f'{lscore:.3f}, ({frame[10,0] / 2 + frame[10,1] / 2})\r', end='', flush=True)

    plines = nlines

    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(test, (l[0], l[1]), (l[2], l[3]), (100), 1, cv2.LINE_AA)
    #         print(f'({l[0]}, {l[1]}), ({l[2]}, {l[3]})')

    cv2.imshow('H', h)

    cv2.imshow('S', s)

    cv2.imshow('T', test)


	# Display the resulting frame
    cv2.imshow('Frame', frame)

    # cv2.moveWindow('Frame', 40,30)


    # cv2.imshow('Thresh', Thresh)
	# define q as the exit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()
