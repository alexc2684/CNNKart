import numpy as np
import os
import png
import tensorflow as tf
from ControllerReader import ControllerReader

IMG_WIDTH = 640
IMG_HEIGHT = 400
IMG_DEPTH = 3
img_dir = "data/frames/frames1/"
dtm_dir = "data/inputs/race1"

def read_images(dirname):
    #read each frame
    #flatten, convert
    #return dataset
    images = []
    for img in os.listdir(dirname):
        if img.endswith(".png"):
            print("Loading: ", img)
            imgdata = png.Reader(filename=dirname+img)
            w, h, pixels, metadata = imgdata.asDirect()
            image = np.vstack(pixels)
            image = np.reshape(image, (w, h, IMG_DEPTH))
            images.append(image)
    return images

#find when A button is initially pressed
def findInputStart(dtm_input):
    i = 0:
    while i < len(dtm_input):
        if dtm_input[0] == 2:
            return i
        i += 1
    return 0

def getData(img_dir, dtm_dir):
    reader = ControllerReader(dtm_dir)
    controllerInputs = reader.readInput()
    frames = read_images(img_dir)
    dtm_start = findInputStart(controllerInputs)
    if dtm_start == 0:
        print("Error: A Button never pressed")
        return
    controllerInputs = controllerInputs[dtm_start:]
    controllerInputs = controllerInputs[:len(controllerInputs) - 2000]
    labels = []
    if len(controllerInputs) > 3*len(frames):
        for i in range(0, len(frames)):
            labels.append(controllerInputs[i*3][4])
    else:
        labels = [controllerInputs[i][4] for i in range(0, len(controllerInputs), 3)]
    print("Frames ", len(frames))
    print("Labels ", len(labels))
    return frames, labels
