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
    while dtm_input[i][0] != 16:
        i += 1
    while dtm_input[i][0] == 16:
        i += 1
    return i

#load all frame data and controller inputs into training set
def load_train_data(img_dir, dtm_dir):
    #init train/validation
    x_train = []
    y_train = []
    frame_data = os.listdir(img_dir)
    frame_data.sort()
    frame_data.remove(".DS_Store")
    inputs = os.listdir(dtm_dir)
    inputs.sort()
    inputs.remove(".DS_Store")
    for img, dtm in zip(frame_data, inputs):
        print(img_dir + "/" + img, dtm_dir + "/" + dtm)
        if os.path.isdir(img_dir + "/" + img):
            reader = ControllerReader(dtm_dir + "/" + dtm)
            controllerInputs = reader.readInput()
            dtm_start = findInputStart(controllerInputs)
            frames = read_images(img_dir + "/" + img + "/")
            if dtm_start == 0:
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
