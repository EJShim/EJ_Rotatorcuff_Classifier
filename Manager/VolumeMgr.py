import mudicom
import vtk
from vtk.util import numpy_support

import numpy as np
import scipy.ndimage


import matplotlib.pyplot as plt

class E_VolumeManager:
    def __init__(self, Mgr):
        self.Mgr = Mgr

        self.m_volumeArray = 0.0
        self.m_bAxial = False;

        #Selected Volume CFGS
        self.m_colorFunction = vtk.vtkColorTransferFunction()
        self.m_opacityFunction = vtk.vtkPiecewiseFunction()
        self.m_scalarRange = [0.0, 1.0]
        self.m_volumeProperty = vtk.vtkVolumeProperty()
        self.m_imageProperty = vtk.vtkImageProperty()
        self.m_imageProperty.SetInterpolationTypeToLinear()



        #Volume
        self.m_volumeMapper = vtk.vtkSmartVolumeMapper()
        self.m_volume = vtk.vtkActor()
        self.m_resliceMapper = [0, 0, 0]
        self.m_resliceActor = [0, 0, 0]

        #Color MAp Volume
        self.m_colorMapMapper = vtk.vtkSmartVolumeMapper()
        self.m_colorMapVolume = vtk.vtkActor()
        self.m_colorMapResliceMapper = [0, 0, 0]
        self.m_colorMapResliceActor = [0, 0, 0]


        self.resolution = 64


        for i in range(3):
            self.m_resliceMapper[i] = vtk.vtkImageSliceMapper()
            self.m_resliceActor[i] = vtk.vtkImageSlice()

            self.m_colorMapResliceMapper[i] = vtk.vtkImageSliceMapper()
            self.m_colorMapResliceActor[i] = vtk.vtkImageSlice()



        #Initialize
        self.SetPresetFunctions(self.Mgr.mainFrm.volumeWidget.GetCurrentColorIndex())
        # self.InitializeRenderFunctions()

    def SetPresetFunctions(self, idx, update = False):

        #Housefield unit : -1024 ~ 3072
        if update == False:
            self.m_colorFunction.RemoveAllPoints()
            self.m_opacityFunction.RemoveAllPoints()

        housefiledRange = 3072 + 1024
        sRange = self.m_scalarRange[1] - self.m_scalarRange[0]

        self.m_imageProperty.SetColorLevel((self.m_scalarRange[1] + self.m_scalarRange[0])/2)
        self.m_imageProperty.SetColorWindow(self.m_scalarRange[1] - self.m_scalarRange[0]-1)


        rangeFactor = sRange / housefiledRange

        if idx == 0: #MIP
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[0], 1.0, 1.0, 1.0)
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[1], 1.0, 1.0, 1.0)

            self.m_opacityFunction.AddPoint(self.m_scalarRange[0], 0.0)
            self.m_opacityFunction.AddPoint(self.m_scalarRange[1], 1.0)

            self.m_volumeProperty.ShadeOff()
            self.m_volumeProperty.SetInterpolationTypeToLinear()

            self.m_volumeMapper.SetBlendModeToMaximumIntensity()

        elif idx == 1: #CT_SKIN
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[0], 0.0, 0.0, 0.0, 0.5, 0.0)
            self.m_colorFunction.AddRGBPoint((-1000 + 1024) / rangeFactor , 0.62, 0.36, 0.18, 0.5, 0.0)
            self.m_colorFunction.AddRGBPoint((-500 + 1024) / rangeFactor , 0.88, 0.60, 0.29, 0.33, 0.45)
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[1], 0.83, 0.66, 1.0, 0.5, 0.0)

            self.m_opacityFunction.AddPoint(self.m_scalarRange[0],0.0, 0.5, 0.0)
            self.m_opacityFunction.AddPoint((-1000 + 1024) / rangeFactor, 0.0, 0.5, 0.0)
            self.m_opacityFunction.AddPoint((-500 + 1024) / rangeFactor, 1.0, 0.33, 0.45)
            self.m_opacityFunction.AddPoint(self.m_scalarRange[1], 1.0, 0.5, 0.0)

            self.m_volumeProperty.ShadeOn()
            self.m_volumeProperty.SetInterpolationTypeToLinear()

            self.m_volumeMapper.SetBlendModeToComposite()


        elif idx == 2: #CT_BONE
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[0], 0.0, 0.0, 0.0, 0.5, 0.0)
            self.m_colorFunction.AddRGBPoint((-16 + 1024) / rangeFactor , 0.73, 0.25, 0.30, 0.49, 0.0)
            self.m_colorFunction.AddRGBPoint((641 + 1024) / rangeFactor , 0.90, 0.82, 0.56, 0.5, 0.0)
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[1], 1.0, 1.0, 1.0, 0.5, 0.0)

            self.m_opacityFunction.AddPoint(self.m_scalarRange[0],0.0, 0.5, 0.0)
            self.m_opacityFunction.AddPoint((-16 + 1024) / rangeFactor, 0.0, 0.49, 0.61)
            self.m_opacityFunction.AddPoint((-641 + 1024) / rangeFactor, 0.72, 0.5, 0.0)
            self.m_opacityFunction.AddPoint(self.m_scalarRange[1], 0.71, 0.5, 0.0)

            self.m_volumeProperty.ShadeOn()
            self.m_volumeProperty.SetInterpolationTypeToLinear()

            self.m_volumeMapper.SetBlendModeToComposite()

        elif idx == 3: #Voxel
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[0], 0.0, 0.0, 1.0)
            self.m_colorFunction.AddRGBPoint((self.m_scalarRange[0] + self.m_scalarRange[1])/2.0, 0.0, 1.0, 0.0)
            self.m_colorFunction.AddRGBPoint(self.m_scalarRange[1], 1.0, 0.0, 0.0)

            self.m_opacityFunction.AddPoint(self.m_scalarRange[0], 0.0)
            self.m_opacityFunction.AddPoint(self.m_scalarRange[1], 1.0)

            self.m_volumeProperty.ShadeOn()
            self.m_volumeProperty.SetInterpolationTypeToLinear()

            self.m_volumeMapper.SetBlendModeToComposite()



    def ImportVolume(self, fileSeries):
        volumeBuffer = []

        for i in range( len(fileSeries) ):
            mu = mudicom.load(fileSeries[i])

            #Load Image
            img = mu.image.numpy
            shape = img.shape
            img = img.reshape(shape[1], shape[0])

            #Initialize Volume Info
            if i == 0:

                #t1? t2?
                reptime = list(mu.find(0x0018, 0x0080))[0].value
                whattime = list(mu.find(0x0018, 0x0081))[0].value

                self.Mgr.SetLog(str(reptime))
                self.Mgr.SetLog(str(whattime))

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
                self.m_bAxial = self.IsAxial(orientation)

                #Pixel Spacing
                pixelSpacing = list(mu.find(0x0028, 0x0030))[0].value
                pixelSpacing = list(map(float, [x.strip() for x in pixelSpacing.split('\\')]))

            volumeBuffer.append(img)


        #Make Volume ARray
        volumeArray = np.asarray(volumeBuffer, dtype=np.uint16)

        #Rotate Volume According to patient coordiante
        renderSpacing = np.array([spacing, pixelSpacing[0], pixelSpacing[1]])


        #Resample to [1,1,1]
        volumeArray, renderSpacing = self.ResampleVolumeData(volumeArray, renderSpacing)

        if self.m_bAxial:
            #Axis(0,2 changes the z-dimension)
            volumeArray = np.rot90(volumeArray, 3, axes=(0,2))
            renderSpacing = [renderSpacing[1], renderSpacing[2], renderSpacing[0]]
        else:
            volumeArray = np.rot90(volumeArray, 3, axes=(1,2))

        self.m_volumeArray = volumeArray


        #Calculate Crop Region Start Position and Dimension(Length)
        # volumeData = volumeArray
        # volumeData, renderSpacing = self.MakeVolumeData(volumeArray, renderSpacing)
        xp = self.Mgr.mainFrm.m_rangeSlider[0].value() / 1000
        yp = self.Mgr.mainFrm.m_rangeSlider[1].value() / 1000

        volumeData = self.MakeVolumeDataWithResampled(volumeArray, xPos = xp, yPos = yp)
        volumeData = (volumeData * 255.0) / np.amax(volumeArray)

        self.m_volumeData = volumeData



        self.AddVolume(volumeData, renderSpacing)
        self.Mgr.PredictObject(volumeData)

    def IsAxial(self, orientation):
        ori = np.round(np.abs(orientation))
        axl =  np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        return np.array_equal(ori, axl)

    def AddClassActivationMap(self, camArray):
        ata_string = camArray.tostring()
        dim = camArray.shape

        imgData = vtk.vtkImageData()
        imgData.SetOrigin([0, 0, 0])
        imgData.SetDimensions(dim[1], dim[2], dim[0])
        imgData.AllocateScalars(vtk.VTK_UNSIGNED_INT, 1);
        imgData.SetSpacing([1.0, 1.0, 1.0])



        for i in range(dim[0]):
            for j in range(dim[2]):
                for k in range(dim[1]):
                    imgData.SetScalarComponentFromDouble(k, j, i, 0, camArray[i][k][j])


        #set Class Activation Map
        colorFunction = vtk.vtkColorTransferFunction()
        opacityFunction = vtk.vtkPiecewiseFunction()
        scalarRange = imgData.GetScalarRange()
        # print("Scalar Range:",scalarRange)
        volumeProperty = vtk.vtkVolumeProperty()

        colorFunction.AddRGBPoint((scalarRange[0] + scalarRange[1])*0.0, 0.0, 0.0, 1.0)
        colorFunction.AddRGBPoint((scalarRange[0] + scalarRange[1])*0.5, 0.0, 1.0, 0.0)
        colorFunction.AddRGBPoint(scalarRange[1], 1.0, 0.0, 0.0)

        opacityFunction.AddPoint((scalarRange[0] + scalarRange[1])*0.0, 0.0)
        opacityFunction.AddPoint(scalarRange[1], 0.05)

        volumeProperty.SetColor(colorFunction)
        volumeProperty.SetScalarOpacity(opacityFunction)
        volumeProperty.ShadeOff()
        volumeProperty.SetInterpolationTypeToLinear()

        self.m_colorMapMapper.SetInputData(imgData)
        self.m_colorMapMapper.SetBlendModeToComposite()

        #Actor
        self.m_colorMapVolume = vtk.vtkVolume()
        self.m_colorMapVolume.SetMapper(self.m_colorMapMapper)
        self.m_colorMapVolume.SetProperty(volumeProperty)
        self.m_colorMapVolume.SetPosition([0, 0, 0])


        lookupTable = vtk.vtkLookupTable()
        lookupTable.SetTableRange(scalarRange)
        lookupTable.SetHueRange(scalarRange)
        lookupTable.SetSaturationRange(scalarRange[0], scalarRange[1])
        lookupTable.SetValueRange(scalarRange)
        lookupTable.Build()


        imageProperty = vtk.vtkImageProperty()
        imageProperty.SetInterpolationTypeToLinear()
        imageProperty.SetLookupTable(lookupTable)
        imageProperty.SetColorLevel((scalarRange[1] + scalarRange[0])/2)
        imageProperty.SetColorWindow(scalarRange[1] - scalarRange[0]-1)
        imageProperty.SetOpacity(0.2)

        #Slice
        for i in range(3):
            self.m_colorMapResliceMapper[i].SetInputData(imgData)
            self.m_colorMapResliceMapper[i].SetOrientation(i)
            self.m_colorMapResliceActor[i].SetMapper(self.m_colorMapResliceMapper[i])
            self.m_colorMapResliceActor[i].SetProperty(imageProperty)

            #Add SLice
            self.Mgr.m_sliceRenderer[i].AddViewProp(self.m_colorMapResliceActor[i])
            self.Mgr.m_sliceRenderer[i].ResetCamera()
            self.Mgr.m_sliceRenderer[i].GetActiveCamera().Zoom(1.5)


        #Add Actor
        self.Mgr.renderer[1].AddVolume(self.m_colorMapVolume)
        self.Mgr.renderer[1].ResetCamera()

        #Set preset
        # self.Mgr.mainFrm.volumeWidget.onChangeIndex(self.Mgr.mainFrm.volumeWidget.GetCurrentColorIndex())


    def AddVolume(self, volumeArray, spacing = [1.0, 1.0, 1.0], origin = [0, 0, 0]):

        data_string = volumeArray.tostring()
        dim = volumeArray.shape
        # print("min: ", np.amin(volumeArray), ", max : ", np.amax(volumeArray))


        imgData = vtk.vtkImageData()
        imgData.SetOrigin(origin)
        imgData.SetDimensions(dim[1], dim[2], dim[0])
        imgData.AllocateScalars(vtk.VTK_UNSIGNED_INT, 1);
        imgData.SetSpacing(spacing[1], spacing[2], spacing[0])

        for i in range(dim[0]):
            for j in range(dim[2]):
                for k in range(dim[1]):
                    imgData.SetScalarComponentFromDouble(k, j, i, 0, volumeArray[i][k][j])

        # floatArray = numpy_support.numpy_to_vtk(num_array=volumeArray.ravel(), deep=True, array_type = vtk.VTK_FLOAT)
        # imgData.GetPointData().SetScalars(floatArray)

        self.m_scalarRange = imgData.GetScalarRange()


        #update Preset OTF

        self.AddVolumeData(imgData, True)

    def AddVolumeData(self, source, type=False):
        self.Mgr.ClearScene()

        # Prepare volume properties.
        self.m_volumeProperty.SetColor(self.m_colorFunction)
        self.m_volumeProperty.SetScalarOpacity(self.m_opacityFunction)


        #Mapper
        if type:
            self.m_volumeMapper.SetInputData(source)

            for i in range(3):
                self.m_resliceMapper[i].SetInputData(source)
        else:
            self.m_volumeMapper.SetInputConnection(source)
            for i in range(3):
                self.m_resliceMapper[i].SetInputConnection(source)

        #Actor
        self.m_volume = vtk.vtkVolume()
        self.m_volume.SetMapper(self.m_volumeMapper)
        self.m_volume.SetProperty(self.m_volumeProperty)
        self.m_volume.SetPosition([0, 0, 0])

        #slice Actor
        for i in range(3):
            self.m_resliceMapper[i].SetOrientation(i)
            self.m_resliceActor[i].SetMapper(self.m_resliceMapper[i])
            self.m_resliceActor[i].SetProperty(self.m_imageProperty)

            #Add SLice
            self.Mgr.m_sliceRenderer[i].AddViewProp(self.m_resliceActor[i])
            self.Mgr.m_sliceRenderer[i].ResetCamera()
            self.Mgr.m_sliceRenderer[i].GetActiveCamera().Zoom(1.5)


            #Update Slider
            minVal = self.m_resliceMapper[i].GetSliceNumberMinValue()
            maxVal = self.m_resliceMapper[i].GetSliceNumberMaxValue()
            self.Mgr.mainFrm.m_sliceSlider[i].setRange(minVal, maxVal)


        #Add Actor
        self.Mgr.renderer[1].AddVolume(self.m_volume)
        self.Mgr.renderer[1].ResetCamera()

        #Set preset
        self.Mgr.mainFrm.volumeWidget.onChangeIndex(self.Mgr.mainFrm.volumeWidget.GetCurrentColorIndex())
        #self.Mgr.Redraw()

    def ForwardSliceImage(self, idx):
        sliceNum = self.m_resliceMapper[idx].GetSliceNumber()
        if sliceNum >= self.m_resliceMapper[idx].GetSliceNumberMaxValue():
            return

        # self.ChangeSliceIdx(idx, sliceNum + 1)
        self.Mgr.mainFrm.m_sliceSlider[idx].setValue(sliceNum + 1)

    def BackwardSliceImage(self, idx):
        sliceNum = self.m_resliceMapper[idx].GetSliceNumber()

        sliceNum = self.m_resliceMapper[idx].GetSliceNumber()
        if sliceNum <= self.m_resliceMapper[idx].GetSliceNumberMinValue():
            return

        # self.ChangeSliceIdx(idx, sliceNum - 1)

        #Set Slider Value
        self.Mgr.mainFrm.m_sliceSlider[idx].setValue(sliceNum - 1)


    def ChangeSliceIdx(self, idx, sliceNum):
        self.m_resliceMapper[idx].SetSliceNumber(sliceNum)
        self.m_colorMapResliceMapper[idx].SetSliceNumber(sliceNum)
        self.Mgr.Redraw2D()

    def ResampleVolumeData(self, volume, spacing):
        #spacing to [1, 1, 1]
        new_spacing = np.array([1, 1, 1])


        resize_factor = spacing / new_spacing
        new_real_shape = volume.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / volume.shape
        new_spacing = spacing / real_resize_factor

        real_resize_factor = np.absolute(real_resize_factor)
        #
        # print(real_resize_factor)

        new_volume = scipy.ndimage.zoom(volume, real_resize_factor, mode='nearest')

        return new_volume, new_spacing

    def MakeVolumeDataWithResampled(self, volume, xPos = 0.5, yPos = 0.5, rot = 0):
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
            rDim = 64
            volume = scipy.ndimage.zoom(volume, ( self.resolution / volume.shape[0], self.resolution /volume.shape[1] , self.resolution / volume.shape[2]), order=5)

            #rotate around y-axis
            volume = np.rot90(volume, rot)

            return volume


    def MakeVolumeData(self, volume, spacing, rot = 0):

        # return volume, spacing

        if np.argmin(volume.shape) == 0:
            channel = spacing[0]
            voxelSize = min([spacing[1], spacing[2]])

            rDim = int(volume.shape[0] * channel / voxelSize)
            xMin = int((volume.shape[1] - rDim) / 4.0)
            yMin = int((volume.shape[2] - rDim) / 1.6)

            volume = volume[:, xMin:,yMin:]
            volume = volume[:, :rDim,:rDim]
        else:
            channel = spacing[2]
            voxelSize = min([spacing[0], spacing[1]])

            rDim = int(volume.shape[2] * channel / voxelSize)
            xMin = int((volume.shape[0] - rDim) / 3.5)
            yMin = int((volume.shape[1] - rDim) / 3.5)

            #Crop Volume
            volume = volume[xMin:,yMin:, :]
            volume = volume[:rDim,:rDim, :]


        #Resample Volume ARray , update spacing info
        volume = scipy.ndimage.zoom(volume, ( self.resolution / volume.shape[0], self.resolution /volume.shape[1] , self.resolution / volume.shape[2]), order=5)
        spacing = [voxelSize, voxelSize, voxelSize]

        #rotate around y-axis
        volume = np.rot90(volume, rot)

        return volume, spacing

    def UpdateVolumeDataCrop(self, xP, yP):
        #if self.m_volumeArray == 0.0: return

        volumeData = self.MakeVolumeDataWithResampled(self.m_volumeArray, xPos = xP, yPos = yP)
        volumeData = (volumeData * 255.0) / np.amax(self.m_volumeArray)

        self.m_volumeData = volumeData

        self.AddVolume(volumeData, [1, 1, 1])
        self.Mgr.PredictObject(volumeData)
