# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
import os

directory = os.fsencode("./new_hacks")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".mp4"): 
        # print(filename)

        clip = VideoFileClip("./new_hacks" + "/" + filename)
        duration = clip.duration
        if(duration > 1):
            clip = clip.subclip(clip.duration - 1, clip.duration).crop(x1=848,y1=428,x2=1072,y2=652)

            clip.write_videofile("./new_hacks" + "/" + "new_" + filename)
