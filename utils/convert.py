import os, sys
import mudicom
import numpy as np
import scipy.ndimage
import random

resolution = 64

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
    volume = scipy.ndimage.zoom(volume, ( resolution / volume.shape[0], resolution /volume.shape[1] , resolution / volume.shape[2]), order=5)

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
            raise Exception('File Extension Error  : ', extension)


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

RAW_DATA_PATH = "/home/ej/data/RCT/18"
saveDir = os.path.join( os.path.dirname(os.path.realpath(__file__)), "../NetworkData/volume" )
ROI_MIN = 4
ROI_MAX = 7
TEST_DATA_RATE = 0.0


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
paths = []
isTrainData = []
num_patients = 0

for className in classes: #Class Directory : RCT and non-RCT

    if className.startswith('.'):
        continue

    classPath =  os.path.join(RAW_DATA_PATH, className)
    subdirs = os.listdir( classPath )
    num_patients += len(subdirs)

    for patient in subdirs:#Patient Directory

        patientDataPath = os.path.join( classPath, patient )

        trainData = False
        if random.random() > TEST_DATA_RATE:
            trainData = True

        if not os.path.isdir(patientDataPath):
            continue
        else:
            dataSeries = os.listdir(patientDataPath)
            for volume in dataSeries: #Series Directory
                volumeDataPath = os.path.join(patientDataPath, volume)


                if not os.path.isdir(volumeDataPath): #if not directory,
                    continue
                else:
                    dicomSeries = os.listdir(volumeDataPath)
                    if len(dicomSeries) < 16 or len(dicomSeries) > 50: #if Number of slice is less than 5,
                        continue

                    paths.append(volumeDataPath)
                    isTrainData.append(trainData)




#Data augmentation according to rotation and ROI clipping
augfac = (ROI_MAX - ROI_MIN)

total = len(paths) * 4 * augfac * augfac -1
current = 0
print("Total expected Training + Test Data :", total)

for idx, path in enumerate(paths):
    out = os.path.normpath(path)
    out = path.split(os.sep)


    for rot in range(4): #Rotation
        for xPos in range(ROI_MIN, ROI_MAX): #ROI Position X
            for yPos in range(ROI_MIN, ROI_MAX): #ROI Position Y
                pathidx = len(out)
                print("(", current, "/", total , ")");
                Anot = "[" + out[pathidx-3] + "][" + out[pathidx-2] + "] ser" + out[pathidx-1] + "rot" + str(rot*90) + "[" + str(xPos) + str(yPos) + "]"

                current += 1;

                clname = int(out[pathidx-3] == 'RCT')
                if not clname == 0 and not clname == 1:
                    print("WTF???")


                #Import Volume
                try:

                    data = ImportVolume(path, xPos/(10), yPos/(10), rot)
                    # if data == None:
                    #     continue



                    if isTrainData[idx]:
                        XData.append(data)
                        yData.append(clname)
                    else: #20% rate Test Set
                        xtData.append(data)
                        ytData.append(clname)
                        ztData.append(Anot)

                except Exception as e:
                    print(e)
                    continue



#Save File
print("Save Data in ", saveDir)

if not(os.path.exists(saveDir) ):
    os.mkdir(saveDir, 0o777)

X = np.asarray(XData)
try:
    if not X.shape[0] == 0:
        print("Saving Training Data :", X.shape[0])
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2], X.shape[3])
        y = np.asarray(yData)

        saveName = str(num_patients) + "patients_rotatorcuff_train_" + str(X.shape[0]) + "_" + str(resolution) + "d.npz"
        trainPath = os.path.join(saveDir, saveName)
        np.savez_compressed( trainPath, features=X, targets=y)
except Exception as e:
    print(e)
    print(X.shape)



XT = np.asarray(xtData)
try:
    if not XT.shape[0] == 0:
        print("Saving Test Data :", XT.shape[0])
        XT = XT.reshape(XT.shape[0], 1, XT.shape[1], XT.shape[2], XT.shape[3])
        YT = np.asarray(ytData)
        ZT = np.asarray(ztData)

        saveName = str(num_patients) + "patients_rotatorcuff_test_" + str(XT.shape[0]) + "_" + str(resolution) + "d.npz"
        testPath = os.path.join(saveDir, saveName)
        np.savez_compressed( testPath, features=XT, targets=YT, names=ZT)

except Exception as e:
    print(e)
    print(XT.shape)


# print("Training Data :" , X.shape[0])
# print("Test Data :", XT.shape[0])

print("Processing Done!")
