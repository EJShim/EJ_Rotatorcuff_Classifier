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
import tensorflow as tf
import network.VRN_64_TF as config_module

#define argument path
file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
weight_path = os.path.join(root_path, "weights", "2block", "epoch49model.ckpt")
model_path = os.path.join(root_path, "data", "TestData.npz")
v_res = 1

class E_Manager:
    def __init__(self, mainFrm):
        self.mainFrm = mainFrm
        self.VolumeMgr = E_VolumeManager(self)
        self.renderer = [0, 0]
        self.m_sliceRenderer = [0, 0, 0]

        self.sess = tf.InteractiveSession()
        self.m_bPred = False

        #Get Features and Target Data
        try:
            data_load = np.load(model_path)
            self.xt = np.asarray(data_load['features'], dtype=np.float32)
            self.yt = np.asarray(data_load['targets'], dtype=np.float32)
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

        for i in range(3):            
            rendererIdx = (i+1)%3
            interactor = E_InteractorStyle2D(self, rendererIdx)            
            # self.mainFrm.m_vtkSliceWidget[i].GetRenderWindow().AddRenderer()            
            self.mainFrm.m_vtkSliceWidget[i].GetRenderWindow().GetInteractor().SetInteractorStyle(interactor)
            interactor.AddRenderer(self.m_sliceRenderer[rendererIdx])
            

        #Initialize
        self.InitObject()        
        self.InitData()
        self.InitNetwork()


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
        

    def InitNetwork(self):
        self.tensor_in, y, self.keep_prob, last_conv = config_module.get_very_shallow_model()
        #Restore Graph
        try:
            saver = tf.train.Saver()
            saver.restore(self.sess, weight_path)            
        except Exception as e:
            self.SetLog("trained graph not found" + str(e))
            self.sess.run(tf.global_variables_initializer())

        y = tf.contrib.layers.flatten(y)
        self.pred_classes = tf.argmax(y, axis=1)
        self.pred_probs = tf.nn.softmax(y)



        gr = tf.get_default_graph()
        last_weight = gr.get_tensor_by_name('fc/kernel:0')
        last_conv = last_conv[0]
        last_weight = last_weight[:,:,:,:,1]
        self.class_activation_map =tf.nn.relu( tf.reduce_sum(tf.multiply(last_weight, last_conv), axis=3))


    def predict_tensor(self, input):
        return self.sess.run([self.pred_classes, self.pred_probs, self.class_activation_map], feed_dict={self.tensor_in:input, self.keep_prob:1.0})
        

    def InitData(self):
        for i in range(len(self.yt)):
            self.mainFrm.m_listWidget.insertItem(i, str(i) + str(".") + str(labels.rt[int(self.yt[i])]))
        

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

        if predict:            
            self.PredictObject(self.xt[idx], int(self.yt[idx]))        

    def ShowScore(self, label, softmax):        
        self.mainFrm.SetProgressScore(softmax[0], label)

    def PredictObject(self, inputData, label = -1):
        if not self.m_bPred:             
            self.VolumeMgr.RemoveClassActivationMap()
            return

        resolution = self.VolumeMgr.resolution
        inputData = np.asarray(inputData.reshape(1, resolution, resolution, resolution, 1), dtype=np.float32)        

        predict_result = self.predict_tensor(inputData)
        pred_class = predict_result[0]
        
        softmax = predict_result[1]
        self.ShowScore(label, softmax)

        #Class Activation Map         
        activation_map = predict_result[2]
        deconv_rate =  64 / activation_map.shape[0]

        activation_map = scipy.ndimage.zoom(activation_map, deconv_rate)

        activation_map = activation_map / 8
        log = "min : " + str(np.amin(activation_map)) + ", max : " + str(np.amax(activation_map))
        self.SetLog(log)
        activation_map *= 255.0
        activation_map = activation_map.astype(int)
        self.VolumeMgr.AddClassActivationMap(activation_map)


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

    def ClearScene(self):        
        self.VolumeMgr.RemoveVolumeData()
        self.VolumeMgr.RemoveClassActivationMap()
        
    def RotateCamera(self):
        camera = self.renderer[1].GetActiveCamera()        
        camera.Azimuth(1)
        camera.SetViewUp(0.0, 1.0, 0.0)

    def SetLog(self, text, error=False):
        QApplication.processEvents() 
        self.mainFrm.statusBar().showMessage(text)

    def PredictROI(self):
        selectedVolume = self.VolumeMgr.m_volumeArray
        shape = selectedVolume.shape
        inputData = np.asarray(selectedVolume.reshape(1, 1, shape[0], shape[1], shape[2]), dtype=np.float32)
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