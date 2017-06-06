
class ControllerReader:
    self.inputDir = None
    self.inputData = None

    def __init__(self, inputDir):
        self.inputDir = inputDir


    def readInput(self): #returns controller data per frame
        data = []
        fileLength = len(f.read())/32 - 8

        if self.inputDir == None or self.inputDir = "":
            print("Error: No input directory specified")
        else:
            with open(self.inputDir, "rb") as f:
                    f.seek(32*8) #skip header
                    for i in range(fileLength):
                        cb = f.read(32)[:8] #only read inputs
                        data.append(cb)
                        print(cb)
                        print(list(cb),"\n")

            self.inputData = data #TODO: change to np.matrix?

            
