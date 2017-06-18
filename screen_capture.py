import numpy as np
import cv2
from mss import mss
from PIL import Image

mon = {'top': 0, 'left': 0, 'width': 2880, 'height': 1800}

sct = mss()

while 1:
    sct.get_pixels(mon)
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    cv2.imshow('test', np.array(img))
    #TODO: feed img into forward prop
    #change resolution?
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
