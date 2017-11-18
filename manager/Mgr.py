import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
import numpy as np
import random
import scipy.ndimage
from time import gmtime, strftime
from PyQt5.QtWidgets import QApplication
from manager.InteractorStyle import E_InteractorStyle
from manager.InteractorStyle import E_InteractorStyle2D
from manager.VolumeMgr import E_VolumeManager
from manager.E_SliceRenderer import *
import matplotlib.pyplot as plt
from data import labels

v_res = 1
#define argument path
file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
weight_path = os.path.join(root_path, "weights", "10-23 09:43:11gpuarray_oct_20_epoch_94.npz")
modelPath = root_path + "/data/TestData.npz"

class E_Manager:
    def __init__(self, mainFrm):
        self.mainFrm = mainFrm

        #Initialize Managers
        self.VolumeMgr = E_VolumeManager(self)
        self.renderer = [0, 0]
        #Ax, Cor, SAg
        self.m_sliceRenderer = [0, 0, 0]

        self.bInitNetowrk = False

        #Test function
        self.predFunc = None        
        self.scoreFunc = None



        #Get Features and Target Data
        try:
            self.xt = np.asarray(np.load(modelPath)['features'], dtype=np.float32)
            self.yt = np.asarray(np.load(modelPath)['targets'], dtype=np.float32)
        except Exception as e:
            self.SetLog(str(e))
            self.xt = []
            self.yt = []

        for i in range(2):
            interactor = E_InteractorStyle(self, i)

            self.renderer[i] = vtk.vtkRenderer()
            self.renderer[i].SetBackground(0.0, 0.0, 0.0)
            self.mainFrm.m_vtkWidget[i].GetRenderWindow().AddRenderer(self.renderer[i])
            self.mainFrm.m_vtkWidget[i].GetRenderWindow().Render()
            self.mainFrm.m_vtkWidget[i].GetRenderWindow().GetInteractor().SetInteractorStyle(interactor)

        for i in range(3):            
            self.m_sliceRenderer[i] = E_SliceRenderer(self,i)
            # self.m_sliceRenderer[i].SetBackground(0.0, 0.0, 0.0)
            # self.m_sliceRenderer[i].GetActiveCamera().ParallelProjectionOn()

            #Set Camera position
            # if i==0:
            #     self.m_sliceRenderer[i].GetActiveCamera().Azimuth(90)
            # elif i==1:
            #     self.m_sliceRenderer[i].GetActiveCamera().Elevation(-90)

        for i in range(3):            
            rendererIdx = (i+1)%3
            interactor = E_InteractorStyle2D(self, rendererIdx)            
            # self.mainFrm.m_vtkSliceWidget[i].GetRenderWindow().AddRenderer()            
            self.mainFrm.m_vtkSliceWidget[i].GetRenderWindow().GetInteractor().SetInteractorStyle(interactor)
            interactor.AddRenderer(self.m_sliceRenderer[rendererIdx])
            



        #Initialize
        self.InitObject()
        self.InitTextActor()
        self.InitData()


    def InitObject(self):
        #Orientatio WIdget
        self.orWidget = [0, 0]
        for i in range(len(self.orWidget)):
            axis = vtk.vtkAxesActor()
            self.orWidget[i] = vtk.vtkOrientationMarkerWidget()
            self.orWidget[i].SetOutlineColor(0.9300, 0.5700, 0.1300)
            self.orWidget[i].SetOrientationMarker(axis)
            self.orWidget[i].SetInteractor(  self.mainFrm.m_vtkWidget[i].GetRenderWindow().GetInteractor() )
            self.orWidget[i].SetViewport(0.0, 0.0, 0.3, 0.3)

        self.Redraw()
        self.Redraw2D()

    def InitTextActor(self):
        self.groundTruthLog = vtk.vtkTextActor()
        self.groundTruthLog.SetInput("Label")
        self.groundTruthLog.SetPosition(10, 60)
        self.groundTruthLog.GetTextProperty().SetFontSize(24)
        self.groundTruthLog.GetTextProperty().SetColor(1.0, 0.0, 0.0)


        self.predLog = vtk.vtkTextActor()
        self.predLog.SetInput("Predicted")
        self.predLog.SetPosition(10, 30)
        self.predLog.GetTextProperty().SetFontSize(24)
        self.predLog.GetTextProperty().SetColor(0.0, 1.0, 0.0)


        self.renderer[1].AddActor2D(self.groundTruthLog)
        self.renderer[1].AddActor2D(self.predLog)
 
                
    def VoxelizeObject(self, source):
        #Transform Polydata around Z-axis
        trans = vtk.vtkTransform()
        trans.RotateWXYZ(-90.0, 0, 0, 1.0)
        transFilter = vtk.vtkTransformPolyDataFilter()
        transFilter.SetTransform(trans)
        transFilter.SetInputConnection(source.GetOutputPort())
        transFilter.Update()

        poly = vtk.vtkPolyData()
        poly.DeepCopy(transFilter.GetOutput())

        #Set Voxel Space Resolution nxnxn
        resolution = self.VolumeMgr.resolution
        bounds = [0, 0, 0, 0, 0, 0]
        center = poly.GetCenter()
        poly.GetBounds(bounds)


        #Get Maximum Boundary Length
        maxB = 0.0
        for i in range(0, 6, 2):
            if abs(bounds[i] - bounds[i+1]) > maxB:
                maxB = abs(bounds[i] - bounds[i+1])

        #Calculate Spacing
        spacingVal = maxB / resolution
        spacing = [spacingVal, spacingVal, spacingVal]

        bounds = [center[0] - resolution * spacing[0] / 2, center[0] + resolution * spacing[0] / 2,center[1] - resolution * spacing[1] / 2, center[1] + resolution * spacing[2] / 2, center[2] - resolution * spacing[2] / 2, center[2] + resolution * spacing[0] / 2]

        imgData = vtk.vtkImageData()
        imgData.SetSpacing(spacing)
        origin = [center[0] - resolution * spacing[0] / 2, center[1] - resolution * spacing[1] / 2, center[2] - resolution * spacing[2] / 2]
        imgData.SetOrigin(origin)

        #Dimensions
        dim = [resolution, resolution, resolution]
        imgData.SetDimensions(dim)
        imgData.SetExtent(0, dim[0]-1, 0, dim[1]-1, 0, dim[2]-1)
        imgData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        for i in range(imgData.GetNumberOfPoints()):
            imgData.GetPointData().GetScalars().SetTuple1(i, 1)

        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(poly)
        pol2stenc.SetOutputOrigin(origin)
        pol2stenc.SetOutputSpacing(spacing)
        pol2stenc.SetOutputWholeExtent(imgData.GetExtent())

        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(imgData)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(0)
        imgstenc.Update()


        scalarData = vtk_to_numpy( imgstenc.GetOutput().GetPointData().GetScalars() )
        self.DrawVoxelArray(scalarData)

        self.PredictObject(scalarData)

    def Redraw(self):
        for i in range(2):
            self.mainFrm.m_vtkWidget[i].GetRenderWindow().Render()
            self.orWidget[i].SetEnabled(1)
    def Redraw2D(self):
        for i in range(3):
            self.m_sliceRenderer[i].GetRenderWindow().Render()
            # self.sliceOrWidget[i].SetEnabled(1)

    def ImportObject(self, path):
        self.SetLog(path)
        filename, file_extension = os.path.splitext(path)

        if file_extension == ".stl":

            #Remove All Actors
            self.ClearScene()

            reader = vtk.vtkSTLReader()
            reader.SetFileName(path)
            reader.Update()


            self.VoxelizeObject(reader)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            self.renderer[0].AddActor(actor)
            self.renderer[0].ResetCamera()
            self.Redraw()

        elif file_extension == ".obj":
            #Remove All Actors
            self.ClearScene()

            reader = vtk.vtkOBJReader()
            reader.SetFileName(path)
            reader.Update()


            self.VoxelizeObject(reader)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            self.renderer[0].AddActor(actor)
            self.renderer[0].ResetCamera()
            self.Redraw()


        else:
            self.SetLog('File Extension Not Supported')

    def SetFunctions(self, pred, color, param):
        self.predFunc = pred
        self.colorMap = color
        self.param_dict = param
        self.bInitNetowrk = True;

    def InitNetwork(self):        
        self.SetLog("Load config and model files..")
        
        model = config_module.get_model()
        scoreModel = score_module.get_model()


        #Compile Functions
        self.SetLog('Compiling Theano Functions..')
        self.predFunc, self.colorMap = function_compiler.make_functions(model)
        self.scoreFunc = function_compiler.make_score_functions(scoreModel)
        print(self.scoreFunc)

        #Load Weights
        metadata, self.param_dict = checkpoints.load_weights(weight_path, model['l_out'])
        metadata2, param_dict2 = checkpoints.load_weights(weight_path, scoreModel['l_out'])

        log = 'Import Completed'
        self.SetLog(log) 

        self.bInitNetowrk = True;

    def InitData(self):

        try:
            zt = np.asarray(np.load(modelPath)['targets'])

            for i in range(len(zt)):

                self.mainFrm.m_listWidget.insertItem(i, str(i) + str(".") + str(labels.rt[int(zt[i])]))
        except Exception as e:
            self.SetLog("Model Path not defined or wrong")
            self.SetLog(str(e))

    def RandomPrediction(self):


        #Get Random
        randIdx = random.randint(0, len(self.xt)-1)

        #Draw Object
        resolution = self.VolumeMgr.resolution
        arr = self.xt[randIdx].reshape(resolution, resolution, resolution)
        self.VolumeMgr.AddVolume(arr)
        self.Redraw2D()

        # self.DrawVoxelArray(xt[randIdx])


        log = labels.rt[int(self.yt[randIdx])]

        #Predict 3D object
        self.PredictObject(arr, log)

    def RenderPreProcessedObject(self, idx, predict=True):
        arr = self.xt[idx][0]
        self.VolumeMgr.AddVolume(arr)


        if self.bInitNetowrk and predict:
            #Predict
            # self.yt = np.asarray(np.load(modelPath)['targets'], dtype=np.float32)
            log = labels.rt[int(self.yt[idx])]

            #Predict 3D object
            self.PredictObject(self.xt[idx], log)

        # self.Redraw()
        # self.Redraw2D()



    def PredictObject(self, inputData, groundTruth = "unknown"):

        #Predict Object
        if self.bInitNetowrk:
            resolution = self.VolumeMgr.resolution
            inputData = np.asarray(inputData.reshape(1, 1, resolution, resolution, resolution), dtype=np.float32)
            #inputData = 4.0 * inputData - 1.0

            
            colorMap = self.colorMap(inputData)
            pred = self.predFunc(inputData)

            predIdx = np.argmax(pred)
            predRate = np.amax(pred)*100.0
            #Show Log
            gtlog = "Label : " + groundTruth
            self.groundTruthLog.SetInput(gtlog)
            log = "Predicted : " + labels.rt[predIdx] + " -> " + str(predRate) + "%"
            self.predLog.SetInput(log)


            #Draw only RCT
            predIdx = 1
            # #Compute Class Activation Map            
            fc1_weight = self.param_dict['fc.W']
            predWeights = fc1_weight[:,predIdx:predIdx+1]
            camsum = np.zeros((colorMap.shape[2], colorMap.shape[3], colorMap.shape[4]))
            for i in range(colorMap.shape[1]):
                camsum = camsum + predWeights[i] * colorMap[0,i,:,:,:]            
            camsum = scipy.ndimage.zoom(camsum, 16)

            log = "min : " + str(np.amin(camsum)) + ", max : " + str(np.amax(camsum))
            self.SetLog(log)
            

            cam_min = np.amin(camsum)
            cam_max = np.amax(camsum)


            #Normalize To 0-255
            tmp = camsum - cam_min
            camsum = tmp / cam_max               
            camsum *= 255.0
            camsum = camsum.astype(int)

            self.VolumeMgr.AddClassActivationMap(camsum)

            # self.Redraw()

        else:
            #$self.SetLog('Network Need to be Initialized')
            return


    def MakeDataMatrix(self, x, intensity):
        return intensity*np.repeat(np.repeat(np.repeat(x[0][0], v_res, axis=0), v_res, axis=1), v_res, axis=2)    

    def DrawVoxelArray(self, arrayBuffer):
        #reshape
        resolution = self.VolumeMgr.resolution
        sample = arrayBuffer.reshape(1, 1, resolution, resolution, resolution)
        dataMatrix = self.MakeDataMatrix( np.asarray(sample, dtype=np.uint8), 255)

        data_string = dataMatrix.tostring()


        dataImporter = vtk.vtkImageImport()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(1)
        dataImporter.SetDataExtent(0, int(dim * v_res)-1, 0, int(dim * v_res)-1, 0, int(dim * v_res)-1)
        dataImporter.SetWholeExtent(0, int(dim * v_res)-1, 0, int(dim * v_res)-1, 0, int(dim * v_res)-1)

        self.VolumeMgr.AddVolumeData(dataImporter.GetOutputPort())
        #Display BoundignBox
        boundingBox = vtk.vtkOutlineFilter()
        boundingBox.SetInputData(dataImporter.GetOutput())

        bbmapper = vtk.vtkPolyDataMapper()
        bbmapper.SetInputConnection(boundingBox.GetOutputPort())

        bbActor = vtk.vtkActor()
        bbActor.SetMapper(bbmapper)
        bbActor.GetProperty().SetColor(1, 0, 0)

        self.renderer[1].AddActor(bbActor)

        self.Redraw()





    def RunGenerativeMode(self):
        self.SetLog("Generative Mode")
        self.SetLog("Reset Renderer")
        self.SetLog("Set View Mode 1view")
        self.SetLog("Run Generative Mode")

    def SyncCamera(self, idx):
        # other = 1
        # if idx == 1: other = 0
        #
        # cam1 = self.renderer[idx].GetActiveCamera()
        # cam2 = self.renderer[other].GetActiveCamera()
        #
        # cam2.DeepCopy(cam1)
        #
        # self.Redraw()
        return

    def ClearScene(self):
        for i in range(2):
            self.renderer[i].RemoveAllViewProps()

            #Add Log Actors
            self.renderer[1].AddActor2D(self.groundTruthLog)
            self.renderer[1].AddActor2D(self.predLog)

        for i in range(3):
            self.m_sliceRenderer[i].RemoveAllViewProps()

    def ClearCAM(self):
        self.renderer[1].RemoveViewProp(self.VolumeMgr.m_colorMapVolume)

        for i in range(3):
            self.m_sliceRenderer[i].RemoveViewProp(self.VolumeMgr.m_colorMapResliceActor[i])

            
            # self.m_sliceRenderer[i].RemoveAllViewProps()

        # self.SetLog(str(self.m_sliceRenderer[0].GetViewProps()))
        
    def RotateCamera(self):
        camera = self.renderer[1].GetActiveCamera()        
        camera.Azimuth(1)
        camera.SetViewUp(0.0, 1.0, 0.0)

    def SetLog(self, text, error=False):
        QApplication.processEvents()

        if error:
            self.mainFrm.m_logWidget.setStyleSheet("color: rgb(255, 0, 255);")
        else:
            self.mainFrm.m_logWidget.setStyleSheet("color: rgb(0, 0, 0);")            
        self.mainFrm.m_logWidget.appendPlainText(text)

    def PredictROI(self):
        selectedVolume = self.VolumeMgr.m_volumeArray
        shape = selectedVolume.shape
        self.SetLog(str(shape))
        inputData = np.asarray(selectedVolume.reshape(1, 1, shape[0], shape[1], shape[2]), dtype=np.float32)

        
        try:            
            scoreVol = self.scoreFunc(inputData)[0][1]
            self.SetLog(str(scoreVol.shape))
        except Exception as e:
            self.SetLog(e)

        camsum = scoreVol
        camsum = scipy.ndimage.zoom(camsum, 16)

        log = "min : " + str(np.amin(camsum)) + ", max : " + str(np.amax(camsum))
        self.SetLog(log)
        

        cam_min = np.amin(camsum)
        cam_max = np.amax(camsum)


        #Normalize To 0-255
        tmp = camsum - cam_min
        camsum = tmp / cam_max               
        camsum *= 255.0
        camsum = camsum.astype(int)

        self.VolumeMgr.AddClassActivationMap(camsum)

        self.SetLog("ROI Prediction")

    def SaveSliceImage(self):

        if self.VolumeMgr.m_resampledVolumeData.any() == None:
            self.SetLog("No Resampled Volume Data is being rendered")
            return

        save_directory = os.path.join(root_path, "humerus_detector", "none_humerus_data")
        data = self.VolumeMgr.m_resampledVolumeData
        
        slice_data = []
        slice_data.append(data[32])
        slice_data.append(np.rot90(data, axes=(0,1))[32])
        slice_data.append(np.rot90(data, axes=(0,2))[32])

        slice_data = np.array(slice_data)
        self.SetLog("slice Data Dim : " + str(slice_data.shape))


        save_directory = os.path.join(root_path, "humerus_detector", "none_humerus_data")
        fname = strftime("%m-%d-%H:%M:%S", gmtime())
        np.savez_compressed(os.path.join(save_directory, fname), features=slice_data)