
from vtk.util import numpy_support
import itk
import vtk
import ctypes

import numpy as np
import scipy.ndimage
from collections import Counter

import matplotlib.pyplot as plt


ImageType = itk.Image[itk.F, 3]

class E_VolumeManager:
    def __init__(self, Mgr):
        self.Mgr = Mgr
                
        self.m_resampledVolumeData = np.array([None])
        self.m_volumeInfo = None
        self.m_selectedIdx = None
        self.m_selectedImage = None        


        # self.m_reverseSagittal = False
        # self.m_reverseAxial = False
        self.m_shoulderSide = 'L'        
        self.m_orientation = 'AXL' ##AXL:1, COR:0, SAG:2

        #Selected Volume CFGS
        self.m_colorFunction = vtk.vtkColorTransferFunction()
        self.m_opacityFunction = vtk.vtkPiecewiseFunction()
        self.m_scalarRange = [0.0, 1.0]
        self.m_volumeProperty = vtk.vtkVolumeProperty()
        self.m_imageProperty = vtk.vtkImageProperty()
        self.m_imageProperty.SetInterpolationTypeToLinear()

        #Volume
        self.volume_data = vtk.vtkImageData()
        self.m_volumeMapper = vtk.vtkSmartVolumeMapper()
        self.m_volume = vtk.vtkVolume()
        self.m_resliceMapper = [0, 0, 0]
        self.m_resliceActor = [0, 0, 0]

        #Color MAp Volume
        self.m_bShowCAM = False
        self.m_bShowVolume = False
        self.m_colorMapMapper = vtk.vtkSmartVolumeMapper()
        self.m_colorMapVolume = vtk.vtkVolume()
        self.m_colorMapResliceMapper = [None, None, None]
        self.m_colorMapResliceActor = [None, None, None]

<<<<<<< HEAD

        self.resolution = 64


=======
        self.resolution = 64

>>>>>>> 155a6a050b620c79b3d0b91c98e00d5d80164a96
        for i in range(3):
            self.m_resliceMapper[i] = vtk.vtkImageSliceMapper()
            self.m_resliceActor[i] = vtk.vtkImageSlice()
            self.m_colorMapResliceMapper[i] = vtk.vtkImageSliceMapper()
<<<<<<< HEAD
            self.m_colorMapResliceActor[i] = vtk.vtkImageSlice()

        
=======
            self.m_colorMapResliceActor[i] = vtk.vtkImageSlice()        
>>>>>>> 155a6a050b620c79b3d0b91c98e00d5d80164a96

        self.m_resliceActor[0].RotateY(90)
        self.m_resliceActor[1].RotateX(-90)
        self.m_colorMapResliceActor[0] .RotateY(90) 
        self.m_colorMapResliceActor[1] .RotateX(-90)

<<<<<<< HEAD


=======
>>>>>>> 155a6a050b620c79b3d0b91c98e00d5d80164a96
        #Initialize
        self.SetPresetFunctions(self.Mgr.mainFrm.volumeWidget.GetCurrentColorIndex())        
        self.InitializeVolumeFunctions()
        self.InitializeClassActivationMap()
        # self.InitializeRenderFunctions()

    def InitializeVolumeFunctions(self):
        #Init Mapper
        self.m_volumeMapper.SetInputData(self.volume_data)
        for i in range(3):
            self.m_resliceMapper[i].SetInputData(self.volume_data)

        # Prepare volume properties.
        self.m_volumeProperty.SetColor(self.m_colorFunction)
        self.m_volumeProperty.SetScalarOpacity(self.m_opacityFunction)

        #Init Actor
        self.m_volume.SetMapper(self.m_volumeMapper)
        self.m_volume.SetProperty(self.m_volumeProperty)
        self.m_volume.SetPosition([0, 0, 0])

        #Init Slice
        for i in range(3):
            self.m_resliceMapper[i].SetOrientation(i)
            self.m_resliceActor[i].SetMapper(self.m_resliceMapper[i])
            self.m_resliceActor[i].SetProperty(self.m_imageProperty)

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
        #Series = 0x0020, 0x0011
        #Instance = 0x0020, 0x0013

        namesGenerator = itk.GDCMSeriesFileNames.New()
        namesGenerator.SetUseSeriesDetails(True)
        namesGenerator.AddSeriesRestriction("0008|0021")
        namesGenerator.SetGlobalWarningDisplay(False)
        namesGenerator.SetDirectory(fileSeries)
        seriesUID = namesGenerator.GetSeriesUIDs()

        serieses = []
        studyDescription = None
        for seriesIdentifier in seriesUID:
            fileNames = namesGenerator.GetFileNames(seriesIdentifier)
            reader = itk.ImageSeriesReader[ImageType].New()
            dicomIO = itk.GDCMImageIO.New()
            reader.SetImageIO(dicomIO)
            reader.SetFileNames(fileNames)
            reader.Update()
            serieses.append(reader)

            if studyDescription == None:
                studyDescription = dicomIO.GetMetaDataDictionary()["0008|1030"]
                

        patient = dict(name=studyDescription, serieses = serieses)
        self.m_volumeInfo = patient
        self.UpdateVolumeTree()
<<<<<<< HEAD

        self.Mgr.ClearScene()
        return
        # ##Will be deprecated very very soon
        # seriesArr = list(map(self.get_volume_info, fileSeries))
        # metaInfo = Counter(tok['series'] for tok in seriesArr)
        

        # serieses = dict()
        # for i in metaInfo:        
        #     serieses[str(i)] = dict(description = '', 
        #                             protocol = '',                                     
        #                             direction='', 
        #                             orientation = '', 
        #                             spacing='', 
        #                             pixelSpacing='',                                     
        #                             data = [])
        # axDir = np.asarray([None])
        # corDir = np.asarray([None])
        # sagDir = np.asarray([None])

        # self.m_reverseSagittal = False
        # self.m_reverseAxial = False
        
        # for series in seriesArr:
        #     datadict =  serieses[ str(series['series']) ]
        #     if datadict['description'] == '':
        #         datadict['description'] = series['seriesDescription']
        #         description = series['seriesDescription'].lower()
                

        #         #Direction
        #         if datadict['direction'] == '':                    
        #             orientationx = np.asarray(series['orientation'])[:3]
        #             orientationy = np.asarray(series['orientation'])[3:]
        #             datadict['direction'] = np.cross(orientationx, orientationy)

        #             self.volumeDirections = [None, None, None]
                
        #         #Orientation
        #         orientation = 'unknown'                                
        #         if not description.find('ax') == -1 or not description.find('tra') == -1:
        #             orientation = 'AXL'
        #             if axDir.any() == None:
        #                 axDir = datadict['direction']
        #         if not description.find('cor') == -1:
        #             orientation = 'COR'
        #             if corDir.any() == None:
        #                 corDir = datadict['direction']
        #         if not description.find('sag') == -1:
        #             orientation = 'SAG'           
        #             if sagDir.any() == None:
        #                 sagDir = datadict['direction']     
        #         datadict['orientation'] = orientation

        #         #Protocol
        #         if not description.find('t1') == -1:
        #             datadict['protocol'] = 'T1'
        #         if not description.find('t2') == -1:
        #             datadict['protocol'] = 'T2'
                    
                                
        #     #Spacing
        #     if datadict['spacing'] == '':
        #         datadict['spacing'] = series['spacing']
        #     if datadict['pixelSpacing'] == '':
        #         datadict['pixelSpacing'] = series['pixelSpacing']
            
        #     datadict['data'].append(series)
        

        
        
        # crossproZ = np.cross(corDir, sagDir)[2]
        # if crossproZ < 0 or crossproZ < 0 and corDir[2] * sagDir[2] < 0:
        #     self.m_reverseSagittal = True
        
        # #Check Shoulder Side
        # if corDir[0] < 0 and corDir[1] < 0:
        #     self.m_reverseAxial = True
        # if corDir[0] > 0 and  corDir[1] > 0:            
        #     self.m_shoulderSide = 'R'
        # else:
        #     self.m_shoulderSide = 'L'
    


        # #Reverse Data List if real direction and annotated direction is reversed
        # for series in seriesArr:
        #     datadict =  serieses[ str(series['series']) ]

        #     if len(datadict['data']) < 2: continue

        #     realDir = np.asarray(datadict['data'][1]['position']) - np.asarray(datadict['data'][0]['position'])
        #     if np.dot(realDir, datadict['direction']) <  0:
        #         datadict['data'].reverse()

        #     # print("Series ", str(series['series'])  ,"length : " ,len(serieses[ str(series['series']) ]['data']))


        # patient = dict(name='patient', serieses = serieses)
        # self.m_volumeInfo = patient
        # self.UpdateVolumeTree()


    def IsAxial(self, orientation):
        ori = np.round(np.abs(orientation))
        axl =  np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        return np.array_equal(ori, axl)
=======
        self.Mgr.ClearScene()
        return
>>>>>>> 155a6a050b620c79b3d0b91c98e00d5d80164a96

    def ToggleClassActivationMap(self, state):        
        if state == 2:
            self.ShowClassActivationMap()            
        else:            
            self.RemoveClassActivationMap()

        self.Mgr.Redraw()
        self.Mgr.Redraw2D()
    
    def ShowClassActivationMap(self):
        
        if self.m_bShowCAM: return
        #Add To Renderer
        for i in range(3):
            rendererIdx = i
            self.Mgr.m_sliceRenderer[i].AddViewProp(self.m_colorMapResliceActor[i])
            self.Mgr.m_sliceRenderer[i].ResetCamera()
            self.Mgr.m_sliceRenderer[i].GetActiveCamera().Zoom(1.5)

        #Add Actor
        self.Mgr.renderer[1].AddVolume(self.m_colorMapVolume)
        self.Mgr.renderer[1].ResetCamera()
        self.m_bShowCAM = True
        
    def RemoveClassActivationMap(self):
        if not self.m_bShowCAM: return
        #Remove From Renderer
        for i in range(3):                
            self.Mgr.m_sliceRenderer[i].RemoveViewProp(self.m_colorMapResliceActor[i])

        #Add Actor
        self.Mgr.renderer[1].RemoveVolume(self.m_colorMapVolume)
        self.m_bShowCAM = False

    def InitializeClassActivationMap(self):        
        self.cam_data = vtk.vtkImageData()
        self.cam_data.SetOrigin([0, 0, 0])
        self.cam_data.SetDimensions(64,64,64,)
        self.cam_data.AllocateScalars(vtk.VTK_UNSIGNED_INT, 1);
        self.cam_data.SetSpacing([1.0, 1.0, 1.0])

        #set Class Activation Map
        cam_color_function = vtk.vtkColorTransferFunction()
        cam_opacity_function = vtk.vtkPiecewiseFunction()
        scalarRange = [0.0, 255.0]        
        cam_volume_property = vtk.vtkVolumeProperty()

        cam_color_function.AddRGBPoint((scalarRange[0] + scalarRange[1])*0.4, 0.0, 0.0, 1.0)
        cam_color_function.AddRGBPoint((scalarRange[0] + scalarRange[1])*0.7, 0.0, 1.0, 0.0)
        cam_color_function.AddRGBPoint(scalarRange[1], 1.0, 0.0, 0.0)

        cam_opacity_function.AddPoint((scalarRange[0] + scalarRange[1])*0.0, 0.3)
        cam_opacity_function.AddPoint(scalarRange[1], 0.3)

        cam_volume_property.SetColor(cam_color_function)
        cam_volume_property.SetScalarOpacity(cam_opacity_function)
        cam_volume_property.ShadeOff()
        cam_volume_property.SetInterpolationTypeToLinear()

        self.m_colorMapMapper.SetInputData(self.cam_data)
        self.m_colorMapMapper.SetBlendModeToMaximumIntensity()

        #Actor        
        self.m_colorMapVolume.SetMapper(self.m_colorMapMapper)
        self.m_colorMapVolume.SetProperty(cam_volume_property)
        self.m_colorMapVolume.SetPosition([0, 0, 0])

        lookupTable = vtk.vtkLookupTable()
        lookupTable.SetTableRange(0.0, 255.0)
        lookupTable.SetHueRange(0.7, 0.0)
        lookupTable.Build()

        imageProperty = vtk.vtkImageProperty()
        imageProperty.SetInterpolationTypeToLinear()
        imageProperty.SetLookupTable(lookupTable)
        imageProperty.SetOpacity(0.3)

        #Slice
        for i in range(3):
            self.m_colorMapResliceMapper[i].SetInputData(self.cam_data)
            self.m_colorMapResliceMapper[i].SetOrientation(i)
            self.m_colorMapResliceActor[i].SetMapper(self.m_colorMapResliceMapper[i])
            self.m_colorMapResliceActor[i].SetProperty(imageProperty)
            
            
        
    def UpdateClassActivationMap(self, camArray):
        #This Function
        self.cam_data.GetPointData().SetScalars(numpy_support.numpy_to_vtk(num_array=camArray.ravel(), deep=True, array_type = vtk.VTK_FLOAT))
        

    def AddClassActivationMap(self, camArray):        
        #This Function
        self.cam_data.GetPointData().SetScalars(numpy_support.numpy_to_vtk(num_array=camArray.ravel(), deep=True, array_type = vtk.VTK_FLOAT))

        if not self.m_bShowCAM: 
            self.ShowClassActivationMap()

        self.Mgr.mainFrm.classCheck.setEnabled(True)



    def AddVolume(self, volumeArray, spacing = [1.0, 1.0, 1.0], origin = [0, 0, 0]):
        floatArray = numpy_support.numpy_to_vtk(num_array=volumeArray.ravel(), deep=True, array_type = vtk.VTK_FLOAT)
        
        dim = volumeArray.shape
        #self.volume_data.AllocateScalars(vtk.VTK_UNSIGNVTK_ED_INT, 1);
        self.volume_data.SetOrigin(origin)
        self.volume_data.SetDimensions(dim[2], dim[1], dim[0])        
        self.volume_data.SetSpacing(spacing)
        self.volume_data.GetPointData().SetScalars(floatArray)
<<<<<<< HEAD
        self.m_scalarRange = self.volume_data.GetScalarRange()
=======
        self.m_scalarRange = self.volume_data.GetScalarRange()        
>>>>>>> 155a6a050b620c79b3d0b91c98e00d5d80164a96
        #Update Slider
        for i in range(3):
            minVal = self.m_resliceMapper[i].GetSliceNumberMinValue()
            maxVal = self.m_resliceMapper[i].GetSliceNumberMaxValue()
            self.Mgr.mainFrm.m_sliceSlider[i].setRange(minVal, maxVal)
        
        #update Preset OTF
        self.ShowVolume()

    def ShowVolume(self):        
        #Add Slice
        for i in range(3):            
            #Add SLice
            rendererIdx = i            
            self.Mgr.m_sliceRenderer[rendererIdx].AddViewProp(self.m_resliceActor[i])
            self.Mgr.m_sliceRenderer[rendererIdx].ResetCamera()
            self.Mgr.m_sliceRenderer[rendererIdx].GetActiveCamera().Zoom(1.5)
            
        #Add Actor
        self.Mgr.renderer[1].AddVolume(self.m_volume)
        self.Mgr.renderer[1].ResetCamera()

        self.m_bShowVolume = True

        #Set preset
        self.Mgr.mainFrm.volumeWidget.onChangeIndex(self.Mgr.mainFrm.volumeWidget.GetCurrentColorIndex(), Update=False)

    def RemoveVolumeData(self):
        if not self.m_bShowVolume: return
        #Add Slice
        for i in range(3):            
            #Add SLice
            rendererIdx = i
            self.Mgr.m_sliceRenderer[rendererIdx].RemoveViewProp(self.m_resliceActor[i])
        #Add Actor
        self.Mgr.renderer[1].RemoveVolume(self.m_volume)        
        self.m_bShowVolume = False
        

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

<<<<<<< HEAD
    def ResampleVolumeData(self, volume, spacing):
        #spacing to [1, 1, 1]
        new_spacing = np.array([1, 1, 1])

        resize_factor = np.amin(spacing)
        print(resize_factor)
        # new_real_shape = volume.shape * resize_factor
        resample_shape = volume.shape
        resample_shape[np.argmin(resample_shape)]*=resize_factor
        new_shape = np.round(resample_shape)
        print(spacing, volume.shape)

        real_resize_factor = new_shape / volume.shape
        new_spacing = spacing / real_resize_factor

        real_resize_factor = np.absolute(real_resize_factor)
        #
        # print(real_resize_factor)

        new_volume = scipy.ndimage.zoom(volume, real_resize_factor, mode='nearest')

        return new_volume, new_spacing

    def MakeVolumeDataWithResampled(self, volume, xPos = 0.5, yPos = 0.5, rot = 0):
            # return volume, spacing
            if np.argmin(volume.shape) == 0: ##COR0                
                rDim = volume.shape[0]

                #Possible X Range
                xMax = volume.shape[1] - rDim
                yMax = volume.shape[2] - rDim

                xMin = int(xMax * xPos)
                yMin = int(yMax * yPos)

                volume = volume[:, xMin:,yMin:]
                volume = volume[:, :rDim,:rDim]

                self.m_decreaseRange = [rDim / xMax, rDim / yMax]
            elif np.argmin(volume.shape) == 1:                
                                        
                rDim = volume.shape[1]

                #Possible X Range
                xMax = volume.shape[2] - rDim
                yMax = volume.shape[0] - rDim
                

                xMin = int(xMax * xPos)
                yMin = int(yMax * yPos)

                volume = volume[xMin:, :, yMin:]
                volume = volume[:rDim, :, :rDim]

                self.m_decreaseRange = [rDim / xMax, rDim / yMax]
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

                self.m_decreaseRange = [rDim / xMax, rDim / yMax]

            
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
        # volumeData = (volumeData * 255.0) / np.amax(self.m_volumeArray)
        self.AddVolume(volumeData, [1, 1, 1])        
        self.Mgr.PredictObject(volumeData)

        self.m_resampledVolumeData = volumeData
=======
    def UpdateVolumeDataCrop(self, xP, yP):
        #Get Selected ITK image
        selected_image = self.m_selectedImage
        crop_position = [0, 0, 0]

        if self.m_orientation == 'AXL':
            crop_position[0] = int(self.crop_position[0] * xP)
            crop_position[2] = int(self.crop_position[2] * yP)            
        elif self.m_orientation == 'COR':
            crop_position[0] = int(self.crop_position[0] * xP)
            crop_position[1] = int(self.crop_position[1] * yP)
        else:
            crop_position[2] = int(self.crop_position[2] * xP)
            crop_position[1] = int(self.crop_position[1] * yP)        


        #Crop + Resample
        resampler = itk.ResampleImageFilter[ImageType, ImageType].New()
        resampler.SetInput(selected_image)        
        resampler.SetOutputStartIndex(crop_position)
        resampler.SetSize([64,64,64])
        resampler.SetOutputSpacing(self.resample_spacing)
        resampler.SetOutputOrigin(selected_image.GetOrigin())
        resampler.SetOutputDirection(selected_image.GetDirection())
        resampler.UpdateLargestPossibleRegion()
        resampler.Update()
        output_image = resampler.GetOutput()

        #Make Array
        volumeBuffer = itk.GetArrayFromImage(output_image)

        #Add Volume
        self.AddVolume(volumeBuffer, [1, 1, 1])
        self.Mgr.PredictObject(volumeBuffer)
        self.Mgr.Redraw()
        self.Mgr.Redraw2D()

        return
>>>>>>> 155a6a050b620c79b3d0b91c98e00d5d80164a96

    def UpdateVolumeTree(self):
        if self.m_volumeInfo == None: return
        self.Mgr.mainFrm.m_treeWidget.updateTree(self.m_volumeInfo)

    def AddSelectedVolume(self, idx):
<<<<<<< HEAD
        self.Mgr.ClearScene()
        self.m_resampledVolumeData = np.array([None])
        self.m_decreaseRange = [1.0, 1.0]
        self.m_selectedIdx = idx

        selected_data = self.m_volumeInfo['serieses'][idx]

        #Adjust Orientation
        # itk_image=selected_data.GetOutput()
=======
        #Clear Scene
        self.Mgr.ClearScene()        
        
        self.m_selectedIdx = idx
        selected_data = self.m_volumeInfo['serieses'][idx]

        #Adjust Orientation        
>>>>>>> 155a6a050b620c79b3d0b91c98e00d5d80164a96
        orienter = itk.OrientImageFilter[ImageType, ImageType].New()
        orienter.UseImageDirectionOn()
        orienter.SetInput(selected_data.GetOutput())
        orienter.Update()
<<<<<<< HEAD
        itk_image = orienter.GetOutput()
        
        #Resample to 64
        dims = itk_image.GetLargestPossibleRegion().GetSize()
        spacing = itk_image.GetSpacing()
        origin = itk_image.GetOrigin()
        direction = itk_image.GetDirection()        
                
        new_size = itk.Size[3]()
        new_spacing = itk.Vector[itk.D,3]()        
        for i in range(3):
            if i==np.argmin(dims):
                new_size[i] = 64
                new_spacing[i] = np.amax(spacing) * (dims[i]/64)
            else:
                new_size[i] = dims[i] 
                new_spacing[i] = spacing[i]
        
        # #Resample Image
        # transform = itk.IdentityTransform[itk.D, 3].New()
        # interpolator = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()
        resampler = itk.ResampleImageFilter[ImageType, ImageType].New()
        resampler.SetInput(itk_image)        
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)        
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputDirection(direction)
        resampler.UpdateLargestPossibleRegion()
        resampler.Update()
        itk_image = resampler.GetOutput()
        
        dims = itk_image.GetLargestPossibleRegion().GetSize()
        spacing = itk_image.GetSpacing()
        origin = itk_image.GetOrigin()
        direction = itk_image.GetDirection()
        
        volumeBuffer = itk.GetArrayFromImage(itk_image)
        volumeArray = np.asarray(volumeBuffer, dtype=np.uint16)
        

=======
        
        #Rescale Image Intensity 0 ~ 255
        normalizer = itk.RescaleIntensityImageFilter[ImageType, ImageType].New()
        normalizer.SetInput(orienter.GetOutput())
        normalizer.SetOutputMinimum(0)
        normalizer.SetOutputMaximum(255)
        normalizer.Update()
        itk_image = normalizer.GetOutput()

        self.m_selectedImage = itk_image
        
        volumeBuffer = itk.GetArrayFromImage(itk_image)        
        
        #Adjust Shoulder Side
>>>>>>> 155a6a050b620c79b3d0b91c98e00d5d80164a96
        # if self.m_shoulderSide == 'L':
        #     volumeArray = np.flip(volumeArray, 2)
        #1,2 = z axes
        #0,2 = y axes
        #0,1 = x axes
        # volumeArray, renderSpacing = self.ResampleVolumeData(volumeArray, spacing)

<<<<<<< HEAD
        #Normalize It
        volumeArray = (volumeArray * 255.0) / np.amax(volumeArray)        
        self.m_volumeArray = volumeArray
        print(volumeArray.shape)

        

        self.AddVolume(volumeArray, spacing)        
        self.Mgr.Redraw()
        self.Mgr.Redraw2D()
        
        return

        #Get Volume Info        
        SeriesData = self.m_volumeInfo['serieses'][idx] 
        data = SeriesData['data']
        description = SeriesData['description'].lower()
        spacing = SeriesData['spacing']
        pixelSpacing = SeriesData['pixelSpacing']
        orientation = SeriesData['orientation']
        direction = SeriesData['direction']

        self.Mgr.mainFrm.m_SeriesNumber.setText(str(idx))
        

        #Rotate Volume According to patient coordiante
        renderSpacing = np.array([spacing, pixelSpacing[0], pixelSpacing[1]])        

        #Make Volume Data
        volumeBuffer = []
        for dic in data:
            volumeBuffer.append(dic['imageData'])


             #Make Volume ARray
        volumeArray = np.asarray(volumeBuffer, dtype=np.uint16)

        if not description.find('t1') == -1:
            self.Mgr.mainFrm.protocolGroup.itemAt(0).widget().setChecked(True)
        elif not description.find('t2') == -1:
            self.Mgr.mainFrm.protocolGroup.itemAt(1).widget().setChecked(True)
            


        if orientation == 'AXL':
            
            volumeArray = np.rot90(volumeArray, axes=(0,1))
            renderSpacing = [renderSpacing[1], renderSpacing[0], renderSpacing[2]]
            
            
            if not self.m_reverseAxial:
                volumeArray = np.rot90(volumeArray,2, axes=(0,2))

            self.Mgr.mainFrm.orientationGroup.itemAt(0).widget().setChecked(True)
        else:
            volumeArray = np.rot90(volumeArray, 2, axes=(1,2))
            self.Mgr.mainFrm.orientationGroup.itemAt(1).widget().setChecked(True)

            if orientation == 'SAG':                            
                if self.m_reverseSagittal:                    
                    volumeArray = np.rot90(volumeArray,3, axes=(0,2))
                else:
                    volumeArray = np.rot90(volumeArray, axes=(0,2))
                renderSpacing = [renderSpacing[2], renderSpacing[1], renderSpacing[0]]
                self.Mgr.mainFrm.orientationGroup.itemAt(2).widget().setChecked(True)

        self.m_orientation = orientation


        if self.m_shoulderSide == 'L':
            volumeArray = np.flip(volumeArray, 2)

        #1,2 = z axes
        #0,2 = y axes
        #0,1 = x axes
        
        volumeArray, renderSpacing = self.ResampleVolumeData(volumeArray, renderSpacing)

        #Normalize It
        volumeArray = (volumeArray * 255.0) / np.amax(volumeArray)
        self.m_volumeArray = volumeArray    

        
        self.AddVolume(volumeArray, renderSpacing)
        #self.Mgr.PredictObject(volumeData)
        self.Mgr.Redraw()
        self.Mgr.Redraw2D()
    
    def GetSelectedSeries(self):
        if self.m_selectedIdx == None:
            self.Mgr.SetLog("No Series selected!")
            return;


        return self.m_volumeInfo['serieses'][self.m_selectedIdx]
         
=======
        

        self.AddVolume(volumeBuffer, itk_image.GetSpacing())        
        self.Mgr.Redraw()
        self.Mgr.Redraw2D()


        #Update Resample Information        
        size = itk_image.GetLargestPossibleRegion().GetSize()
        spacing = itk_image.GetSpacing()
        # direction = itk_image.GetDirection()        
        # origin = itk_image.GetOrigin()
        
        #Get Volume Thickness
        length_of_cube = np.amax(spacing) * np.amin(size)
        new_spacing = length_of_cube / 64

        #New Size and Spacing        
        self.resample_spacing = itk.Vector[itk.D,3]([new_spacing, new_spacing, new_spacing])
        
        #Calculate Crop Position && Save Orientation                 
        self.crop_position =  [0,0,0]   
        orientation = np.argmin(volumeBuffer.shape)
        if orientation == 1:
            self.m_orientation = 'AXL'
            self.crop_position[0] = (size[0]*spacing[0]-length_of_cube)/new_spacing
            self.crop_position[2] = (size[2]*spacing[2]-length_of_cube)/new_spacing
        elif orientation == 0:
            self.m_orientation = 'COR'
            self.crop_position[0] = (size[0]*spacing[0]-length_of_cube)/new_spacing
            self.crop_position[1] = (size[1]*spacing[1]-length_of_cube)/new_spacing
        else:
            self.m_orientation = 'SAG'
            self.crop_position[2] = (size[2]*spacing[2]-length_of_cube)/new_spacing
            self.crop_position[1] = (size[1]*spacing[1]-length_of_cube)/new_spacing
        
        return
>>>>>>> 155a6a050b620c79b3d0b91c98e00d5d80164a96
