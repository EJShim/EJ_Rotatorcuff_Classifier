import os, sys
import mudicom
import numpy as np
import scipy.ndimage

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

    fileList = os.listdir(dataPath)

    for i in range( len(fileList) ):
        bAxial = False

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

    return volumeData



##Main Function
RAW_DATA_PATH = "/home/ej/data/RCT"

#Import path
classes = os.listdir(RAW_DATA_PATH)
XData = []
yData = []

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
                    if len(dicomSeries) < 10 or len(dicomSeries) > 50: #if Number of slice is less than 5,
                        print("slice number error")
                        continue


                    print( "Processing [", classes[className] , "] Data : [", subdirs[patient], "] -->", dataSeries[volume] );

                    #Import Volume
                    try:
                        data = ImportVolume(volumeDataPath, 0.5, 0.5, 0)
                        if data == None:
                            continue

                        XData.append(data)
                        yData.append(className)
                    except Exception:
                        continue

X = np.asarray(XData)
X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2], X.shape[3])
y = np.asarray(yData)
print("Processing Done!")


#Save File
saveDir = os.path.join( os.path.dirname(os.path.realpath(__file__)), "../NetworkData/volume" )
print("Save Data in ", saveDir)

if not(os.path.exists(saveDir) ):
    os.mkdir(saveDir, 0o777)


dataPath = os.path.join(saveDir, "rotatorcuff_train-TEST.npz")
np.savez_compressed( dataPath, features=X, targets=y )

#Chcek -
xt = np.load(dataPath)['features']
yt = np.load(dataPath)['targets']


print(xt.shape)
print(yt.shape)
