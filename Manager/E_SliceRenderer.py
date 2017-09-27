import vtk
import numpy as np
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

        
        #Add Center Position
        point_bottom = (point0 + point1) / 2.0
        point_top = (point2 + point3) / 2.0
        point_left = (point0 + point3) / 2.0
        point_right = (point1 + point2) / 2.0

        centerLinePoints = vtk.vtkPoints()
        centerLinePoints.SetNumberOfPoints(4)
        centerLinePoints.SetPoint(0, point_bottom)
        centerLinePoints.SetPoint(1, point_top)
        centerLinePoints.SetPoint(2, point_left)
        centerLinePoints.SetPoint(3, point_right)
        
        centerLines = vtk.vtkCellArray()
        centerLines.InsertNextCell(2)
        centerLines.InsertCellPoint(0)
        centerLines.InsertCellPoint(1)
        
        centerLines.InsertNextCell(2)
        centerLines.InsertCellPoint(2)
        centerLines.InsertCellPoint(3)

        centerLinePoly = vtk.vtkPolyData()
        centerLinePoly.SetPoints(centerLinePoints)
        centerLinePoly.SetLines(centerLines)

        centerLineMapper = vtk.vtkPolyDataMapper()
        centerLineMapper.SetInputData(centerLinePoly)
        centerLineMapper.Update()

        centerLineActor = vtk.vtkActor()
        centerLineActor.SetMapper(centerLineMapper)
        centerLineActor.GetProperty().SetColor([0.0, 1.0, 1.0])

        self.AddActor(centerLineActor)


        
    

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
        
        