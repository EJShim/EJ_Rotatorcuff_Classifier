import os, sys
import mudicom
import numpy as np
import scipy.ndimage
import random

def ResampleVolumeData(volume, spacing):
    #spacing to [1, 1, 1]
    new_spacing = np.array([1, 1, 1])

    resize_factor = spacing / new_spacing
    new_real_shape = volume.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / volume.shape
    real_resize_factor = np.absolute(real_resize_factor)

    new_spacing = spacing / real_resize_factor

    new_volume = scipy.ndimage.zoom(volume, real_resize_factor, mode='nearest')

    return new_volume, new_spacing

def IsAxial(orientation):
    ori = np.round(np.abs(orientation))
    axl =  np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    return np.array_equal(ori, axl)




def MakeVolumeDataWithResampled(volume, xPos, yPos, rot):
    # return volume, spacing

    if np.argmin(volume.shape) == 0:
        rDim = volume.shape[0]

        #Possible X Range
        xMax = volume.shape[1] - rDim
        yMax = volume.shape[2] - rDim

        xMin = int(xMax * xPos)
        yMin = int(yMax * yPos)

        volume = volume[:, xMin:,yMin:]
        volume = volume[:, :rDim,:rDim]
    else:
        rDim = volume.shape[2]

        #Possible X, Y Range
        xMax = volume.shape[0] - rDim
        yMax = volume.shape[1] - rDim

        xMin = int(xMax * xPos)
        yMin = int(yMax * yPos)

        #Crop Volume
        volume = volume[xMin:,yMin:, :]
        volume = volume[:rDim,:rDim, :]


    #Resample Volume ARray , update spacing info
    rDim = 32
    volume = scipy.ndimage.zoom(volume, ( rDim / volume.shape[0], rDim /volume.shape[1] , rDim / volume.shape[2]), order=5)

    #rotate around y-axis
    volume = np.rot90(volume, rot)

    return volume


def ImportVolume(dataPath, xPos = 0.5, yPos = 0.5, rot = 0):
    volumeBuffer = []
    bAxial = False


    fileList = os.listdir(dataPath)
    fileList.sort()

    for i in range( len(fileList) ):
        filePath = os.path.join(dataPath,fileList[i])
        extension = os.path.splitext(filePath)[1]


        if not extension == ".dcm":
            print("file Extension error : ", extension)
            return None

        mu = mudicom.load(  filePath  )

        #Load Image
        img = mu.image.numpy
        shape = img.shape
        img = img.reshape(shape[1], shape[0])

        #Initialize Volume Info
        if i == 0:
            #Spacing
            spacing = 1.0;
            if len(list(mu.find(0x0018, 0x0088))) > 0:
                spacing = float(list(mu.find(0x0018, 0x0088))[0].value)

            #Position
            position = list(mu.find(0x0020, 0x0032))[0].value

            #Orientation
            orientation = list(mu.find(0x0020, 0x0037))[0].value
            orientation = list(map(float, [x.strip() for x in orientation.split('\\')]))
            #Check if Axial
            bAxial = IsAxial(orientation)

            #Pixel Spacing
            pixelSpacing = list(mu.find(0x0028, 0x0030))[0].value
            pixelSpacing = list(map(float, [x.strip() for x in pixelSpacing.split('\\')]))

        volumeBuffer.append(img)


    #Make Volume ARray
    volumeArray = np.asarray(volumeBuffer, dtype=np.uint16)

    #Rotate Volume According to patient coordiante
    renderSpacing = np.array([spacing, pixelSpacing[0], pixelSpacing[1]])


    #Resample to [1,1,1]
    volumeArray, renderSpacing = ResampleVolumeData(volumeArray, renderSpacing)

    if bAxial:
        #Axis(0,2 changes the z-dimension)
        volumeArray = np.rot90(volumeArray, 3, axes=(0,2))
        renderSpacing = [renderSpacing[1], renderSpacing[2], renderSpacing[0]]

    else:
        volumeArray = np.rot90(volumeArray, 3, axes=(1,2))


    #Calculate Crop Region Start Position and Dimension(Length)
    volumeData = MakeVolumeDataWithResampled(volumeArray,  xPos, yPos, rot)


    #Normalize VolumeData
    volumeData = (volumeData * 255.0) / np.amax(volumeData)


    return volumeData



##Main Functionls

RAW_DATA_PATH = "/home/ej/data/RCT/Train"
saveDir = os.path.join( os.path.dirname(os.path.realpath(__file__)), "../NetworkData/volume" )


#Import path
classes = os.listdir(RAW_DATA_PATH)

#For Training Set
XData = []
yData = []

#For Test Set
xtData = []
ytData = []
ztData = []

print("convert RCT and non-RCT data from", RAW_DATA_PATH, ",with ", classes)

for className in range(len(classes)): #Class Directory : RCT and non-RCT
    classPath =  os.path.join(RAW_DATA_PATH, classes[className])
    subdirs = os.listdir( classPath )

    for patient in range(len(subdirs)):#Patient Directory
        patientDataPath = os.path.join( classPath, subdirs[patient] )

        if not os.path.isdir(patientDataPath):
            print(patientDataPath, ": Not a proper DIR")
            continue
        else:
            # print( "Processing [", classes[className] , "] Data : ", subdirs[patient] );
            dataSeries = os.listdir(patientDataPath)

            for volume in range(len(dataSeries)): #Series Directory
                volumeDataPath = os.path.join(patientDataPath, dataSeries[volume])


                if not os.path.isdir(volumeDataPath): #if not directory,
                    continue
                else:
                    dicomSeries = os.listdir(volumeDataPath)
                    if len(dicomSeries) < 16 or len(dicomSeries) > 50: #if Number of slice is less than 5,
                        print("slice number error")
                        continue



                    for rot in range(4): #Rotation
                        for xPos in range(2, 8): #ROI Position X
                            for yPos in range(2, 8): #ROI Position Y
                                print( "Processing [", classes[className] , "] Data : [", subdirs[patient], "] -->", dataSeries[volume], "rot ", rot*90, xPos, yPos );
                                Anot = "[" + classes[className] + "]/[" + subdirs[patient] + "] ser" + dataSeries[volume] + "rot" + str(rot*90)
                                #Import Volume
                                try:
                                    data = ImportVolume(volumeDataPath, xPos/10, yPos/10, rot)
                                    if data == None:
                                        continue



                                    if random.random() > 0.2: #80% rate Training Set
                                        XData.append(data)
                                        yData.append(className)
                                    else: #20% rate Test Set
                                        xtData.append(data)
                                        ytData.append(className)
                                        ztData.append(Anot)

                                except Exception:
                                    continue

X = np.asarray(XData)
X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2], X.shape[3])
y = np.asarray(yData)


XT = np.asarray(xtData)
XT = XT.reshape(XT.shape[0], 1, XT.shape[1], XT.shape[2], XT.shape[3])
YT = np.asarray(ytData)
ZT = np.asarray(ztData)

print("Processing Done!")


#Save File
print("Save Data in ", saveDir)

if not(os.path.exists(saveDir) ):
    os.mkdir(saveDir, 0o777)


trainPath = os.path.join(saveDir, "rotatorcuff_train.npz")
np.savez_compressed( trainPath, features=X, targets=y)

testPath = os.path.join(saveDir, "rotatorcuff_test.npz")
np.savez_compressed( testPath, features=XT, targets=YT, names=ZT)


#Chcek -
xTrain = np.load(trainPath)['features']
xTest = np.load(testPath)['features']
print("Train Data:", xTrain.shape)
print("Test Data:", xTest.shape)
