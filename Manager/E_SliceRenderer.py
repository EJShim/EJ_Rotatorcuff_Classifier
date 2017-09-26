import vtk
import math

class E_SliceRenderer(vtk.vtk.vtkRenderer):
    def __init__(self, mgr, idx = 0):        
        view = ['SAG', 'AXL', 'COR']
        self.idx = idx
        self.viewType = view[idx]
        self.lineColor = [0.0, 0.0, 0.0]
        self.lineColor[(idx+2)%3] = 1.0

        self.Mgr = mgr

        self.SetBackground(0.0, 0.0, 0.0)
        self.GetActiveCamera().SetPosition(0.0, 0.0, 100.0)
        self.GetActiveCamera().ParallelProjectionOn()

    def AddGuide(self, bounds = [0.0, 0.0, 0.0]):
        
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
        
        point0 = [0.0, 0.0, bounds[2]]
        point1 = [bounds[0], 0.0, bounds[2]]
        point2 = [bounds[0], bounds[1], bounds[2]]
        point3 = [0.0, bounds[1], bounds[2]]

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
        
        