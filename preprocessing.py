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
            print("Loading:", img)
            imgdata = png.Reader(filename=dirname+img)
            w, h, pixels, metadata = imgdata.asDirect()
            image = np.vstack(pixels)
            image = np.reshape(image, (w, h, IMG_DEPTH))
            images.append(image)
    return images

#find when A button is initially pressed
def findInputStart(dtm_input):
    i = 0
    while dtm_input[i][0] != 16 and i < len(dtm_input):
        i += 1
    while dtm_input[i][0] == 16 and i < len(dtm_input):
        i += 1
    if i == len(dtm_input) - 1:
        return -1
    return i

def findFiles(folder):
    paths = []
    for path in folder:
        if os.path.isfile(path):
            paths.append(path)
    return paths

#load all frame data and controller inputs into training set
def load_train_data(img_dir, dtm_dir):
    #init train/validation
    x_train = []
    y_train = []
    frame_data = os.listdir(img_dir)
    frame_data.sort()
    for path in frame_data:
        if path == ".DS_Store":
            frame_data.remove(path)
    inputs = os.listdir(dtm_dir)
    inputs.sort()
    for path in inputs:
        if path == ".DS_Store":
            inputs.remove(path)
    for path in findFiles(frame_data):
        frame_data.remove(path)
    for img, dtm in zip(frame_data, inputs):
        print(img_dir + "/" + img, dtm_dir + "/" + dtm)
        if os.path.isdir(img_dir + "/" + img):
            reader = ControllerReader(dtm_dir + "/" + dtm)
            controllerInputs = reader.readInput()
            dtm_start = findInputStart(controllerInputs)
            frames = read_images(img_dir + "/" + img + "/")
            if dtm_start == -1:
                print("Error: Y Button never pressed")
                return
            controllerInputs = controllerInputs[dtm_start:]
            controllerInputs = controllerInputs[:len(controllerInputs) - 2000]
            labels = []
            if len(controllerInputs) > 3*len(frames):
                for i in range(0, len(frames)):
                    labels.append(controllerInputs[i*3][4])
            else:
                labels = [controllerInputs[i][4] for i in range(0, len(controllerInputs), 3)]
                frames = frames[:len(labels)]
            x_train.extend(frames)
            y_train.extend(labels)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train
