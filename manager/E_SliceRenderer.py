import vtk
import numpy as np
import math

class E_SliceRenderer(vtk.vtkRenderer):
    def __init__(self, mgr, idx = 0):        
        view = ['SAG', 'AXL', 'COR']
        self.idx = idx
        self.viewType = view[idx]
        self.lineColor = [0.0, 0.0, 0.0]
        self.lineColor[(idx+2)%3] = 1.0

        self.centerPos = np.array([0.0, 0.0])
        self.selectedPos = np.array([0.0, 0.0])
        self.bounds = [0.0, 0.0, 0.0]
        self.m_bShowGuide = False

    
        self.Mgr = mgr

        self.SetBackground(0.0, 0.0, 0.0)
        self.GetActiveCamera().SetPosition(0.0, 0.0, 100.0)
        self.GetActiveCamera().ParallelProjectionOn()

        self.Initialize()        
        
    def Initialize(self):
        self.centerLineActor = vtk.vtkActor()
        self.polygonActor = vtk.vtkActor()
        self.selectedPositionActor = vtk.vtkActor()

        self.centerLineActor.GetProperty().SetColor([0.0, 0.4, 0.8])
        self.polygonActor.GetProperty().SetColor(self.lineColor)
        self.selectedPositionActor.GetProperty().SetColor([0.7, 0.7, 0.0])
        
        #Initialize Center selection
        self.selected_position_poly = vtk.vtkPolyData()
        self.selected_position_mapper = vtk.vtkPolyDataMapper()
        self.selected_position_mapper.SetInputData(self.selected_position_poly)
        self.selectedPositionActor.SetMapper(self.selected_position_mapper)

        

    def AddGuide(self, bounds = [0.0, 0.0, 0.0]):                

        self.bounds = bounds
        
        selectedOrienation = self.Mgr.VolumeMgr.m_orientation

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
        self.polygonActor.SetMapper(polygonMapper)        

        
        #Add Center Position
        point_bottom = (point0 + point1) / 2.0
        point_top = (point2 + point3) / 2.0
        point_left = (point0 + point3) / 2.0
        point_right = (point1 + point2) / 2.0

        self.centerPos = np.array([point_top[0], point_left[1]])

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
        
        self.centerLineActor.SetMapper(centerLineMapper)        


        self.AddActor(self.polygonActor)        
        self.AddActor(self.centerLineActor)

    def RemoveGuide(self):
        self.RemoveActor(self.polygonActor)
        self.RemoveActor(self.selectedPositionActor)
        self.RemoveActor(self.centerLineActor)
        self.m_bShowGuide = False
        

    def UpdateSelectedPosition(self, position = [0, 0]):        
        point_bottom = [position[0], 0.0, self.bounds[2]]
        point_top = [position[0], self.bounds[1], self.bounds[2]]
        point_left = [0.0, position[1], self.bounds[2]]
        point_right = [self.bounds[0], position[1], self.bounds[2]]

        self.selectedPos = np.array([position[0], position[1]])


        #Center Line
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(4)
        points.SetPoint(0, point_bottom)
        points.SetPoint(1, point_top)
        points.SetPoint(2, point_left)
        points.SetPoint(3, point_right)

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)        
        lines.InsertNextCell(2)
        lines.InsertCellPoint(2)
        lines.InsertCellPoint(3)

        self.selected_position_poly.SetPoints(points)
        self.selected_position_poly.SetLines(lines)
        

        # self.centerLineMapper.Update()
        self.GetRenderWindow().Render()


        if not self.m_bShowGuide:
            self.AddActor(self.selectedPositionActor)
            self.m_bShowGuide = True

    def CalculateDiff(self):                
        
        #Only When it is preprocessed, selected original orientation case..        
        selectedOrientation = self.Mgr.VolumeMgr.m_orientation
        if not selectedOrientation == self.viewType:
            log = "Selected from " + selectedOrientation + "view"
            self.Mgr.SetLog(log)
            return

        if self.Mgr.VolumeMgr.m_resampledVolumeData.any() == None:
            self.Mgr.SetLog("Select From Original Image")

            pos = [0, 0]
            pos[0] = int((self.selectedPos[0] / self.bounds[0]) * 800.0)
            pos[1] = int((self.selectedPos[1] / self.bounds[1]) * 800.0)
            if selectedOrientation == 'SAG':
                pos = [pos[1], pos[0]]

            self.Mgr.SetLog(str(pos))


            self.Mgr.mainFrm.m_rangeSlider[0].setSliderPosition(pos[1])
            self.Mgr.mainFrm.m_rangeSlider[1].setSliderPosition(pos[0])

            return

        decreaseRange = self.Mgr.VolumeMgr.m_decreaseRange

        diff = self.selectedPos - self.centerPos        
        diff[0] = int((diff[0] / self.bounds[0]) * decreaseRange[1] * 1000.0)
        diff[1] = int((diff[1] / self.bounds[1]) * decreaseRange[0] * 1000.0)
        
        if selectedOrientation == 'SAG':
            diff = [diff[1], diff[0]]
                            
        curX = self.Mgr.mainFrm.m_rangeSlider[0].value()
        curY = self.Mgr.mainFrm.m_rangeSlider[1].value()            
    
        self.Mgr.mainFrm.m_rangeSlider[0].setSliderPosition(curX + diff[1])
        self.Mgr.mainFrm.m_rangeSlider[1].setSliderPosition(curY + diff[0])
    

    def AddViewProp(self, prop):
        self.RemoveGuide()
        bounds = [prop.GetMaxXBound(), prop.GetMaxYBound(), prop.GetMaxZBound()]
        super(E_SliceRenderer, self).AddViewProp(prop)                
        self.AddGuide(bounds)
        
        