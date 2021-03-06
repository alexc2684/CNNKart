import numpy as np

class ControllerReader:
    inputDir = None
    inputData = None

    def __init__(self, inputDir):
        self.inputDir = inputDir


    def readInput(self): #returns controller data per frame
        data = []
        if self.inputDir == None or self.inputDir == "":
            print("Error: No input directory specified")
        else:
            print("Reading controller input data")
            with open(self.inputDir, "rb") as f:
                    fileLength = len(f.read())/32 - 8
                    f.seek(32*8) #skip header
                    for i in range(int(fileLength)):
                        cb = f.read(32)[:8] #only read inputs
                        cb = [ord(i) for i in cb]
                        data.append(cb)
                        print(cb)
            data = np.array(data)
            self.inputData = data
            return data
