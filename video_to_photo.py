import cv2
import os
import logging


def read_video(input_filename):
    return cv2.VideoCapture(input_filename)


def save_frame(count, sec, vid_cap, output_directory):
    vid_cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, frame = vid_cap.read()

    if hasFrames:
        name = os.path.join(os.getcwd(), output_directory, "frame" + str(count) + ".png")
        cv2.imwrite(name, frame)

    return hasFrames


def get_frames(input_filename, output_directory, frameRate):
    """
    Capture images from a video at every (i.e, 5 sec, 5 mn, etc.)
    :param frameRate: time we want to capture the images, in seconds.
    """
    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    except OSError:
        logging.error('Error creating directory')

    sec = 0
    count = 1
    vid_cap = read_video(input_filename)
    success = save_frame(count, sec, vid_cap, output_directory)

    while success:
        count += 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = save_frame(count, sec, vid_cap, output_directory)


if __name__ == "__main__":
    directory = "./hacks_data/"
    i = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mp4"): 
            print(os.path.join(directory, filename))
            get_frames(os.path.join(directory, filename), ("./hacks_data_nn/" + str(i) + "/"), 1/60)
            i += 1