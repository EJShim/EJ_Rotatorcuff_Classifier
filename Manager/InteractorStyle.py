import vtk

class E_InteractorStyle(vtk.vtkInteractorStyleSwitch):
    def __init__(self, Manager, idx):
        self.Mgr = Manager;
        self.idx = idx

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



    def OnMouseWheelForward(self, obj, event):
        self.Mgr.VolumeMgr.ForwardSliceImage(self.idx)

    def OnMouseWheelBackward(self, obj, event):
        self.Mgr.VolumeMgr.BackwardSliceImage(self.idx)
