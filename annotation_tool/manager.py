import sys,os
import vtk
from vtk.util import numpy_support
import numpy as np



file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
model_path = os.path.join(root_path, 'data', 'TestData.npz')
tmp_path = os.path.join(file_path, 'tmp.npz')

#Load Data
data = np.load(model_path)


#Load Temp Annotation
if os.path.isfile(tmp_path):
    tmp_data = np.load(tmp_path)['data']            
else:
    print('tmp file not exists')
    tmp_data = [None] * 200
    np.savez_compressed(tmp_path, data=tmp_data)


#Initialize Renderers
renderers = []


#Initialize Props
color_func = vtk.vtkColorTransferFunction()
opac_func = vtk.vtkPiecewiseFunction()
volume_prop = vtk.vtkVolumeProperty()
volume_mapper = vtk.vtkSmartVolumeMapper()
volume_actor =  vtk.vtkVolume()

volume_prop.SetColor(color_func)
volume_prop.SetScalarOpacity(opac_func)
volume_actor.SetMapper(volume_mapper)
volume_actor.SetProperty(volume_prop)
volume_actor.SetPosition([0, 0, 0])


image_prop = vtk.vtkImageProperty()
slice_mapper = []
slice_actor = []
for i in range(3):
    slice_mapper.append(vtk.vtkImageSliceMapper())
    slice_actor.append(vtk.vtkImageSlice())

    slice_mapper[i].SetOrientation(i)
    slice_actor[i].SetMapper(slice_mapper[i])
    slice_actor[i].SetProperty(image_prop)



def get_tmp_data():
    return tmp_data

def save_tmp_data(tmp_arr):
    np.savez_compressed(tmp_path, data=tmp_arr)

def get_volume(idx = None):
    volume_list = data['features']
    if idx == None:
        return volume_list
    else:
        return np.reshape(volume_list[idx], (64, 64, 64))

def get_renderers(idx = None):    

    if len(renderers) == 0:    
        for i in range(0, 4):
            renderers.append(vtk.vtkRenderer())
            


    if idx == None:
        return renderers
    else:
        return renderers[idx]

def add_volume(vol_array):    
    dim = vol_array.shape
    imgData = vtk.vtkImageData()
    imgData.SetDimensions(dim[2], dim[1], dim[0])
    imgData.SetOrigin([0, 0, 0])    
    imgData.SetSpacing([1.0,1.0,1.0])
    imgData.AllocateScalars(vtk.VTK_UNSIGNED_INT, 1);
    floatArray = numpy_support.numpy_to_vtk(num_array=vol_array.ravel(), deep=True, array_type = vtk.VTK_FLOAT)
    imgData.GetPointData().SetScalars(floatArray)
    
    #Clear Scene
    clear_scene()
    
    #Replace Image Data
    volume_mapper.SetInputData(imgData)

    #slice Actor
    for i in range(3):
        slice_mapper[i].SetInputData(imgData)        

        renderers[i+1].AddViewProp(slice_actor[i])
        renderers[i+1].ResetCamera()
        renderers[i+1].GetActiveCamera().Zoom(1.5)


    

    #Add Actor
    renderers[0].AddVolume(volume_actor)
    renderers[0].ResetCamera()

    redraw()

def redraw():
    for renderer in renderers:        
        renderer.GetRenderWindow().Render()        

def clear_scene():
    for renderer in renderers:
        renderer.RemoveAllViewProps()



class E_InteractorStyle(vtk.vtkInteractorStyleSwitch):
    def __init__(self):           
        self.renderer = None

        #Style to
        self.SetCurrentStyleToTrackballCamera()
        self.GetCurrentStyle().AddObserver("MouseMoveEvent", self.MouseMoveEvent)



    def MouseMoveEvent(self, obj, event):
        self.GetCurrentStyle().OnMouseMove()


class E_InteractorStyle2D(vtk.vtkInteractorStyleImage):
    def __init__(self):        

        self.AddObserver("MouseWheelForwardEvent", self.OnMouseWheelForward)
        self.AddObserver("MouseWheelBackwardEvent", self.OnMouseWheelBackward)    

         
    def OnMouseWheelForward(self, obj, event):        
        print("for")

    def OnMouseWheelBackward(self, obj, event):
        print("back")
        



