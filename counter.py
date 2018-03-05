
import sys
import glob
import numpy as np
import random


numNone = 0
numPartial = 0
numSmall = 0
numMed = 0
numLarge = 0
numMessive = 0

numAXL = 0
numCOR = 0
numSAG = 0

listFile = list(glob.iglob('D:/data/Shoulder_rename/**/*.npz', recursive = True))
random.shuffle(listFile)

total = len(listFile)
cur = 0

features = []
targets = []

testFeatures = []
testTargets = []

print("number of Lists : ", total)
for filename in listFile:
    cur += 1
    
    data = dict(np.load(filename))

    orientation = data['orientation']
    status = data['status']

    

    if orientation == "AXL":        
        numAXL += 1
    elif orientation == "COR":
        numCOR += 1
    elif orientation == "SAG":        
        numSAG += 1

    if status == 'None':
        numNone += 1
        targets.append(0)
    elif status == 'Small':
        continue
        numSmall += 1
        targets.append(1)
    elif status == 'Medium':
        numMed += 1
        targets.append(1)
    elif status == 'Large':
        numLarge += 1
        targets.append(2)
    elif status == 'Massive':
        numMessive += 1
        targets.append(2)
    else:
        continue
        print("Partial Tear not Included")
        # numPartial += 1
        # targets.append(1)
    features.append(data['data'])

    log =  "\rProcessing.... : " + str((cur / total)*100) + "%"
    sys.stdout.write(log)
    sys.stdout.flush()


print("\n None : ", numNone)
print(" Partial : ", numPartial)
print(" Small : ", numSmall)
print(" Medium : ", numMed)
print(" Large : ", numLarge)
print(" Massive : ", numMessive)
print("=======orinetation------------")
print("AXL : ", numAXL)
print("COR : ", numCOR)
print("SAG : ", numSAG)


test_features = features[:200]
test_targets = targets[:200]

train_features = features[200:]
train_targets = targets[200:]


train_features = np.array(train_features)
train_features = train_features.reshape(train_features.shape[0], 1, train_features.shape[1], train_features.shape[2], train_features.shape[3])

test_features = np.array(test_features)
test_features = test_features.reshape(test_features.shape[0], 1, test_features.shape[1], test_features.shape[2], test_features.shape[3])




print("Train Data_3class : ", train_features.shape)
print("Test Data_3class : ", test_features.shape)


np.savez_compressed('./data/TrainData_nonesmall_3cl', features=train_features, targets=train_targets)
np.savez_compressed("./data/TestData_nonesmall_3cl", features=test_features, targets=test_targets)
