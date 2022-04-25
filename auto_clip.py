import cv2
import numpy as np
import sys
import os

# global vars
fc = 0

nlines = 0
plines = 0

decCount = 0

pois = []


assert len(sys.argv) == 3, "needs file input and destination"

inFile = sys.argv[1]
outDir = sys.argv[2]

#inure outDir ends with /
outDir = outDir + ("/" if outDir[-1] != "/" else "")

assert os.path.isfile(inFile), "input file doesn't exist"
assert os.path.isdir(outDir), "output directory doesn't exist"

print(f'Input File: {inFile}')
print(f'Output Dir: {outDir}')
print("")



# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture(inFile)



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

    # shown to user
    view = frame[70:300, 1800:1919]
    # area of video to look analyze
    frame = frame[70:300, 1907:1910]



    # print(nlines)

    nlines = countLines(frame)

    print(f'Kills on Screen: {nlines}, Kills Counted: {len(pois)}\t\t\r', end='', flush=True)


    decCount += 1

    if nlines < plines:
        decCount = 0

    if nlines > plines and decCount > 10:
        # print("Increase")
        pois.append(fc)

    plines = nlines


	# Display the resulting frame
    cv2.imshow('Frame', frame)

    cv2.imshow('View', view)



	# define q as the exit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    fc += 1




# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()


# prints for refrence
print("\n")
print(pois)
print(f'\n----------\nSaving {len(pois)} Clip{"s" if len(pois) != 1 else ""}')

#
# Temp to assert not too many clips recorded
#
assert len(pois) < 33

# how to save a frame modified from video_to_photos
def save_frames(file_name, out_dir, start=0, end=-1):
    cap = cv2.VideoCapture(file_name)

    assert cap, "path to file must be valid"

    if not os.path.exists(out_dir):
            os.makedirs(out_dir)


    if end == -1:
        end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for fno in range(start, end):
        success, frame = cap.read()
        assert success, "frame not read correctly"

        frame = frame[428:652, 848:1072]

        name = os.path.join(os.getcwd(), out_dir, "frame" + str(fno + 1) + ".png")
        cv2.imwrite(name, frame)

    
    cap.release()


# find number of folder to save in
subfolders = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
max = 0

for folder in subfolders:
    if folder.isnumeric() and int(folder) > max:
        max = int(folder)



# save all pois
for x in range(len(pois)):
    assert pois[x] >= 55

    print(f'Saving Clip {x + 1}\t\r', end='', flush=True)

    save_frames(inFile, f'{outDir}{max + x + 1}/', pois[x] - 55, pois[x] + 5)

print("All Clips Saved!\t\t\t\t")
