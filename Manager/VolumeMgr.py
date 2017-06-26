import mudicom
import vtk
from vtk.util import numpy_support

import numpy as np
import scipy.ndimage


import matplotlib.pyplot as plt

class E_VolumeManager:
    def __init__(self, Mgr):
        self.Mgr = Mgr

        self.m_volumeArray = None
        self.m_bAxial = False;

        #Selected Volume CFGS
        self.m_colorFunction = vtk.vtkColorTransferFunction()
        self.m_opacityFunction = vtk.vtkPiecewiseFunction()
        self.m_scalarRange = [0.0, 1.0]
        self.m_volumeProperty = vtk.vtkVolumeProperty()
        self.m_imageProperty = vtk.vtkImageProperty()
        self.m_imageProperty.SetInterpolationTypeToLinear()

        self.m_volumeMapper = vtk.vtkSmartVolumeMapper()
        self.m_volume = vtk.vtkActor()

        self.m_resliceMapper = [0, 0, 0]
        self.m_resliceActor = [0, 0, 0]



        for i in range(3):
            self.m_resliceMapper[i] = vtk.vtkImageSliceMapper()
            self.m_resliceActor[i] = vtk.vtkImageSlice()



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


        #Add To Renderer
        # print("Volume Array Dim : ", volumeArray.shape)
        # print("Processed Volume DAta : ", volumeData.shape)
        self.AddVolume(volumeData, renderSpacing)

    def IsAxial(self, orientation):
        ori = np.round(np.abs(orientation))
        axl =  np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        return np.array_equal(ori, axl)


    def AddVolume(self, volumeArray, spacing = [1.0, 1.0, 1.0], origin = [0, 0, 0]):
        data_string = volumeArray.tostring()
        dim = volumeArray.shape

        imgData = vtk.vtkImageData()
        imgData.SetOrigin(origin)
        imgData.SetDimensions(dim[1], dim[2], dim[0])
        imgData.AllocateScalars(vtk.VTK_UNSIGNED_INT, 1);
        imgData.SetSpacing(spacing[1], spacing[2], spacing[0])

        #
        # for i in range(volumeArray.size):
        #
        #     y = i % dim[2]
        #     x = int(i / dim[2]) % dim[1]
        #     z = int(i / (dim[1] * dim[2]))
        #
        #     # print(z, ", ", x , ",", y)
        #     imgData.SetScalarComponentFromDouble(x, y, z, 0, volumeArray[z][x][y])

        for i in range(dim[0]):
            for j in range(dim[2]):
                for k in range(dim[1]):
                    imgData.SetScalarComponentFromDouble(k, j, i, 0, volumeArray[i][k][j])

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


            #Update Slider
            minVal = self.m_resliceMapper[i].GetSliceNumberMinValue()
            maxVal = self.m_resliceMapper[i].GetSliceNumberMaxValue()
            self.Mgr.mainFrm.m_sliceSlider[i].setRange(minVal, maxVal)



        self.Mgr.Redraw2D()

        #Add Actor
        self.Mgr.renderer[1].AddVolume(self.m_volume)
        self.Mgr.renderer[1].ResetCamera()

        #Set preset
        self.Mgr.mainFrm.volumeWidget.onChangeIndex(self.Mgr.mainFrm.volumeWidget.GetCurrentColorIndex())
        self.Mgr.Redraw()

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
        self.Mgr.Redraw2D()

    def ResampleVolumeData(self, volume, spacing):
        #spacing to [1, 1, 1]
        new_spacing = np.array([1, 1, 1])

        resize_factor = spacing / new_spacing
        new_real_shape = volume.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / volume.shape
        new_spacing = spacing / real_resize_factor

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
            rDim = 32
            volume = scipy.ndimage.zoom(volume, ( rDim / volume.shape[0], rDim /volume.shape[1] , rDim / volume.shape[2]), order=5)

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
        rDim = 32
        volume = scipy.ndimage.zoom(volume, ( rDim / volume.shape[0], rDim /volume.shape[1] , rDim / volume.shape[2]), order=5)
        spacing = [voxelSize, voxelSize, voxelSize]

        #rotate around y-axis
        volume = np.rot90(volume, rot)

        return volume, spacing

    def UpdateVolumeDataCrop(self, xP, yP):
        if self.m_volumeArray == None: return

        volumeData = self.MakeVolumeDataWithResampled(self.m_volumeArray, xPos = xP, yPos = yP)

        self.AddVolume(volumeData, [1, 1, 1])
