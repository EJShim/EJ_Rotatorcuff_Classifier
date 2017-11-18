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
selected_renderer = [0]


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
volume_prop.ShadeOff()
volume_prop.SetInterpolationTypeToLinear()
volume_mapper.SetBlendModeToMaximumIntensity()


image_prop = vtk.vtkImageProperty()
slice_mapper = []
slice_actor = []
for i in range(3):
    slice_mapper.append(vtk.vtkImageSliceMapper())
    slice_mapper[i].BorderOn()
    slice_actor.append(vtk.vtkImageSlice())


    slice_mapper[i].SetOrientation(i)
    slice_actor[i].SetMapper(slice_mapper[i])
    slice_actor[i].SetProperty(image_prop)



def get_tmp_data():
    return tmp_data

def save_tmp_data():
    np.savez_compressed(tmp_path, data=tmp_data)

def get_volume(idx = None):
    volume_list = data['features']
    if idx == None:
        return volume_list
    else:
        return np.reshape(volume_list[idx], (64, 64, 64))

def get_renderers(idx = None):    

    if len(renderers) == 0:    
        renderers.append(vtk.vtkRenderer())
        for i in range(1, 4):
            renderers.append(E_SliceRenderer(idx = i-1))
            


    if idx == None:
        return renderers
    else:
        return renderers[idx]

def set_preset(scalar_range):

    image_prop.SetColorLevel((scalar_range[1] + scalar_range[0])/2)
    image_prop.SetColorWindow(scalar_range[1] - scalar_range[0]-1)

    color_func.RemoveAllPoints()
    opac_func.RemoveAllPoints()

    color_func.AddRGBPoint(scalar_range[0], 1.0, 1.0, 1.0)
    color_func.AddRGBPoint(scalar_range[1], 1.0, 1.0, 1.0)

    opac_func.AddPoint(scalar_range[0], 0.0)
    opac_func.AddPoint(scalar_range[1], 1.0)    

def add_volume(vol_array):    
    dim = vol_array.shape
    imgData = vtk.vtkImageData()
    imgData.SetDimensions(dim[2], dim[1], dim[0])
    imgData.SetOrigin([0, 0, 0])    
    imgData.SetSpacing([1.0,1.0,1.0])
    imgData.AllocateScalars(vtk.VTK_UNSIGNED_INT, 1);
    floatArray = numpy_support.numpy_to_vtk(num_array=vol_array.ravel(), deep=True, array_type = vtk.VTK_FLOAT)
    imgData.GetPointData().SetScalars(floatArray)

    set_preset(imgData.GetScalarRange())
    
    #Clear Scene
    clear_scene()
    
    #Replace Image Data
    volume_mapper.SetInputData(imgData)

    #slice Actor
    for i in range(3):
        slice_mapper[i].SetInputData(imgData)        
        slice_mapper[i].SetSliceNumber(32)

        renderers[i+1].AddViewProp(slice_actor[i])
        renderers[i+1].ResetCamera()
        renderers[i+1].GetActiveCamera().Zoom(1.5)

        renderers[0].AddViewProp(slice_actor[i])


    

    #Add Actor
    renderers[0].AddVolume(volume_actor)
    renderers[0].ResetCamera()

    redraw()

def forward_slice(idx):
    slice_num = slice_mapper[idx].GetSliceNumber()
    if slice_num >= slice_mapper[idx].GetSliceNumberMaxValue():
        return
    
    change_slice(idx, slice_num+1)

def backward_slice(idx):
    slice_num = slice_mapper[idx].GetSliceNumber()
    if slice_num <= slice_mapper[idx].GetSliceNumberMinValue():
        return
    
    change_slice(idx, slice_num-1)

def change_slice(idx, slice_number):    
    slice_mapper[idx].SetSliceNumber(slice_number)        
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

        self.GetCurrentStyle().AddObserver("LeftButtonPressEvent", self.OnLeftButtonPressed)
        self.GetCurrentStyle().AddObserver("MouseMoveEvent", self.MouseMoveEvent)


    def OnLeftButtonPressed(self, obj, event):
        self.GetCurrentStyle().OnLeftButtonDown()
        selected_renderer[0] = 0

    def MouseMoveEvent(self, obj, event):
        self.GetCurrentStyle().OnMouseMove()


class E_InteractorStyle2D(vtk.vtkInteractorStyleImage):
    def __init__(self, idx = 0):
        self.idx = idx
        
        self.AddObserver("LeftButtonPressEvent", self.OnLeftButtonPressed)
        self.AddObserver("MouseWheelForwardEvent", self.OnMouseWheelForward)
        self.AddObserver("MouseWheelBackwardEvent", self.OnMouseWheelBackward) 

    def OnLeftButtonPressed(self, obj, event):
        self.OnLeftButtonDown()
        selected_renderer[0] = self.idx+1

    def OnMouseWheelForward(self, obj, event):        
        forward_slice(self.idx)
        selected_renderer[0] = self.idx+1

    def OnMouseWheelBackward(self, obj, event):
        backward_slice(self.idx)
        selected_renderer[0] = self.idx+1


class E_SliceRenderer(vtk.vtkRenderer):
    def __init__(self, idx = 0):        
        view = ['SAG', 'AXL', 'COR']
        self.idx = idx
        self.viewType = view[idx]
        self.lineColor = [0.0, 0.0, 0.0]
        self.lineColor[(idx+2)%3] = 1.0

        self.centerPos = np.array([0.0, 0.0])
        self.selectedPos = np.array([0.0, 0.0])

        self.selectedPositionActor = None
        self.bounds = [0.0, 0.0, 0.0]


        self.SetBackground(0.0, 0.0, 0.0)
        self.GetActiveCamera().SetPosition(0.0, 0.0, 100.0)
        self.GetActiveCamera().ParallelProjectionOn()

        self.InitSelectedPosition()
        
    def InitSelectedPosition(self):        

        self.selectedPositionActor = vtk.vtkActor()        
        self.selectedPositionActor.GetProperty().SetColor([1.0, 0.0, 1.0])
        
        self.AddActor(self.selectedPositionActor)

    def AddGuide(self, bounds = [0.0, 0.0, 0.0]):
        self.bounds = bounds

        
        #ADD TEXT
        txt = vtk.vtkTextActor()
        txt.SetInput(self.viewType)        
        txt.SetPosition(10, 10)
        txt.GetTextProperty().SetFontSize(24)

        
        txt.GetTextProperty().SetColor(self.lineColor)
        self.AddActor2D(txt)

        #ADD Outer Line   
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(4)
        
        point0 = np.array([0.0, 0.0, bounds[2]])
        point1 = np.array([bounds[0], 0.0, bounds[2]])
        point2 = np.array([bounds[0], bounds[1], bounds[2]])
        point3 = np.array([0.0, bounds[1], bounds[2]])
        

        points.SetPoint(0, point0)
        points.SetPoint(1, point1)
        points.SetPoint(2, point2)
        points.SetPoint(3, point3)

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(5)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        lines.InsertCellPoint(2)
        lines.InsertCellPoint(3)
        lines.InsertCellPoint(0)        

        polygon = vtk.vtkPolyData()
        polygon.SetPoints(points)
        polygon.SetLines(lines)

        polygonMapper = vtk.vtkPolyDataMapper()
        polygonMapper.SetInputData(polygon)
        polygonMapper.Update()

        polygonActor = vtk.vtkActor()
        polygonActor.SetMapper(polygonMapper)
        polygonActor.GetProperty().SetColor(self.lineColor)

        self.AddActor(polygonActor)
    

    def AddViewProp(self, prop):
        rotateProp = vtk.vtkImageSlice()
        rotateProp.SetMapper(prop.GetMapper())
        rotateProp.SetProperty(prop.GetProperty())

        if self.viewType == 'AXL':
            rotateProp.RotateX(-90)        
        elif self.viewType == 'SAG':
            rotateProp.RotateY(90)

        bounds = [rotateProp.GetMaxXBound(), rotateProp.GetMaxYBound(), rotateProp.GetMaxZBound()]        
        

        super(E_SliceRenderer, self).AddViewProp(rotateProp)
        self.AddGuide(bounds)


