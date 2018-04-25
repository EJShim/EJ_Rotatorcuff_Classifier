import vtk

class E_InteractorStyle(vtk.vtkInteractorStyleSwitch):
    def __init__(self, Manager, idx):
        self.Mgr = Manager;
        self.idx = idx
        self.renderer = None



        #Style to
        self.SetCurrentStyleToTrackballCamera()
        self.GetCurrentStyle().AddObserver("MouseMoveEvent", self.MouseMoveEvent)



    def MouseMoveEvent(self, obj, event):
        self.GetCurrentStyle().OnMouseMove()


class E_InteractorStyle2D(vtk.vtkInteractorStyleImage):
    def __init__(self, Manager, idx):
        self.Mgr = Manager;
        self.idx = idx

        self.AddObserver("MouseWheelForwardEvent", self.OnMouseWheelForward)
        self.AddObserver("MouseWheelBackwardEvent", self.OnMouseWheelBackward)
        self.AddObserver("LeftButtonPressEvent", self.OnLeftButtonPressed)
        self.AddObserver("LeftButtonReleaseEvent", self.OnLeftButtonReleased)
        self.AddObserver("RightButtonPressEvent", self.OnRightButtonPressed)
        self.AddObserver("RightButtonReleaseEvent", self.OnRightButtonReleased)
        
        

    def AddRenderer(self, renderer):
        self.renderer = renderer
        self.GetInteractor().GetRenderWindow().AddRenderer(renderer)
        self.GetInteractor().Render()
        



    def OnMouseWheelForward(self, obj, event):        
        self.Mgr.VolumeMgr.ForwardSliceImage(self.idx)

    def OnMouseWheelBackward(self, obj, event):
        self.Mgr.VolumeMgr.BackwardSliceImage(self.idx)

    def OnLeftButtonPressed(self, obj, event):
        
        if self.renderer ==None: return                
        position = self.GetInteractor().GetEventPosition()
        print(position)
        picker = vtk.vtkPropPicker()
        picker.Pick(position[0], position[1], 0, self.renderer)
        self.renderer.UpdateSelectedPosition(picker.GetPickPosition())
        

        self.Mgr.mainFrm 

        

    def OnLeftButtonReleased(self, obj, event):
        ha = 0
        
    def OnRightButtonPressed(self, obj, event):
        self.renderer.CalculateDiff()

    def OnRightButtonReleased(self, obj, event):
        ha = 0
        
