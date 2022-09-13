import glob
from PIL import Image
import numpy as np

def make_gif(fold):
    fname="t{0:.2f}.jpg"
    times = np.hstack((np.arange(0,10,0.1), np.arange(10,11,0.1)))
    frames = []
    for t in times:
        frames.append(Image.open(fold+fname.format(t)))
    frame_one = frames[0]
    frame_one.save("NT.gif", format="GIF", append_images=frames, save_all=True, duration=1000, loop=0)


if __name__=='__main__':
    make_gif("GIF_IMAGES/")