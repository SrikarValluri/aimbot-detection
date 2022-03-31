# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
import os

directory = os.fsencode("./hacks")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".mp4"): 
        # print(filename)

        clip = VideoFileClip("./hacks" + "/" + filename)
        duration = clip.duration
        if(duration > 1):
            clip = clip.subclip(clip.duration - 1, clip.duration).crop(x1=0,y1=0,x2=224,y2=224)

            clip.write_videofile("./hacks" + "/" + "new_" + filename)
