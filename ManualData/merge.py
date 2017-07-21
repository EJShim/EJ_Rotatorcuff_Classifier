import os
import numpy as np


curPath = os.path.dirname(os.path.realpath(__file__))


features = None
targets = None

for directory in os.listdir(curPath):

    path = os.path.join(curPath, directory)

    if os.path.isdir(path):

        for fileName in os.listdir(path):
            filePath = os.path.join(path, fileName)

            if features == None:
                features = np.load(filePath)['features']
                targets = np.load(filePath)['targets']
            else:
                features = np.concatenate((features, np.load(filePath)['features']))
                targets = np.concatenate((targets, np.load(filePath)['targets']))


savePath = os.path.join(curPath, "Merged_BlackWhite")
np.savez_compressed(savePath, features=features, targets=targets)
