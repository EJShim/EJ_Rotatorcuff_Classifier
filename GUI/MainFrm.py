from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import sys, os
from Manager.Mgr import E_Manager
from GUI.VolumeRenderingWidget import E_VolumeRenderingWidget



curPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.abspath(os.path.join(curPath, os.pardir))
iconPath = rootPath + "/icons"

class E_MainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(E_MainWindow, self).__init__(parent)

        self.setWindowTitle("EJ ModelNet Project")

        #Central Widget
        self.m_centralWidget = QWidget()
        self.setCentralWidget(self.m_centralWidget)

        #vtk Renderer Widget
        self.m_vtkWidget = [0, 0]
        for i in range(2):
            self.m_vtkWidget[i] = QVTKRenderWindowInteractor();


        self.m_vtkSliceWidget = [0,0,0]
        self.m_sliceSlider = [0, 0, 0]
        for i in range(3):
            #Slice View
            self.m_vtkSliceWidget[i] = QVTKRenderWindowInteractor();

            #Slice Image Slider
            self.m_sliceSlider[i] = QSlider(Qt.Horizontal)
            self.m_sliceSlider[i].setRange(0, 0)
            self.m_sliceSlider[i].setSingleStep(1)

            self.m_sliceSlider[i].rangeChanged.connect(self.onSliderRangeChanged)
            self.m_sliceSlider[i].valueChanged.connect(self.onSliderValueChanged)


        self.m_sliceViewWidget = QWidget()


        #Initialize

        self.InitToolbar()
        self.InitCentralWidget()
        self.InitSliceViewWidget()
        self.InitManager()


    def InitToolbar(self):
        #ToolBar
        toolbar = QToolBar()

        self.addToolBar(toolbar)


        mainTab = QTabWidget()
        toolbar.addWidget(mainTab)


        objectToolbar = QToolBar();
        objectToolbar.setIconSize(QSize(58, 58))
        objectToolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        mainTab.addTab(objectToolbar, "3D Objects")

        #Import Object Action
        importAction = QAction(QIcon(iconPath + "/051-cmyk.png"), "Import Object", self)
        importAction.triggered.connect(self.onImportObject)
        objectToolbar.addAction(importAction)

        #Import Volume addAction
        volumeAction = QAction(QIcon(iconPath + "/051-document.png"), "Import Volume", self)
        volumeAction.triggered.connect(self.onImportVolume)
        objectToolbar.addAction(volumeAction)
        objectToolbar.addSeparator()

        self.volumeWidget = E_VolumeRenderingWidget()
        objectToolbar.addWidget(self.volumeWidget)

        objectToolbar.addSeparator()


        checkWidget = QWidget()
        objectToolbar.addWidget(checkWidget)
        checkLayout = QVBoxLayout()
        checkWidget.setLayout(checkLayout)

        meshViewCheck = QCheckBox("Mesh View")
        meshViewCheck.setCheckState(2)
        meshViewCheck.stateChanged.connect(self.onMeshViewState)
        checkLayout.addWidget(meshViewCheck)

        objectToolbar.addSeparator()
        cropWidget = QWidget()
        objectToolbar.addWidget(cropWidget)
        cropLayout = QVBoxLayout()
        cropWidget.setLayout(cropLayout)

        self.m_rangeSlider = [0, 0]

        self.m_rangeSlider[0] = QSlider(Qt.Horizontal)
        self.m_rangeSlider[0].setRange(0.0, 1000)
        self.m_rangeSlider[0].setSingleStep(1)
        self.m_rangeSlider[0].setSliderPosition( 500 )
        cropLayout.addWidget(self.m_rangeSlider[0])
        self.m_rangeSlider[0].valueChanged.connect(self.onRangeSliderValueChanged)

        self.m_rangeSlider[1] = QSlider(Qt.Horizontal)
        self.m_rangeSlider[1].setRange(0.0, 1000)
        self.m_rangeSlider[1].setSingleStep(1)
        self.m_rangeSlider[1].setSliderPosition( 500 )
        cropLayout.addWidget(self.m_rangeSlider[1])
        self.m_rangeSlider[1].valueChanged.connect(self.onRangeSliderValueChanged)




        volumeViewCheck = QCheckBox("Volume View")
        volumeViewCheck.setCheckState(2)
        volumeViewCheck.stateChanged.connect(self.onVolumeViewState)
        checkLayout.addWidget(volumeViewCheck)


        sliceViewCheck = QCheckBox("Slice View")
        sliceViewCheck.setCheckState(0)
        sliceViewCheck.stateChanged.connect(self.onSliceViewState)
        checkLayout.addWidget(sliceViewCheck)


        networkToolbar = QToolBar();
        networkToolbar.setIconSize(QSize(58, 58))
        networkToolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        mainTab.addTab(networkToolbar, "VRN")

        self.trainAction = QAction(QIcon(iconPath + "/051-pantone-2.png"), "Initialize Network", self)
        self.trainAction.triggered.connect(self.onInitNetwork)
        networkToolbar.addAction(self.trainAction)

        predAction = QAction(QIcon(iconPath + "/051-programming.png"), "Predict Random", self)
        predAction.triggered.connect(self.onRandomPred)
        networkToolbar.addAction(predAction)

    def InitSliceViewWidget(self):
        layout = QVBoxLayout()


        for i in range(3):
            layout.addWidget(self.m_vtkSliceWidget[i])
            layout.addWidget(self.m_sliceSlider[i])

        #hide initialize
        self.m_sliceViewWidget.setLayout(layout)
        self.m_sliceViewWidget.hide()


    def InitCentralWidget(self):
        MainLayout = QHBoxLayout()
        self.m_centralWidget.setLayout(MainLayout)


        for i in range(2):
            self.m_vtkWidget[i]
            MainLayout.addWidget(self.m_vtkWidget[i])

        MainLayout.addWidget(self.m_sliceViewWidget)

        #dock widget
        dockwidget = QDockWidget("Log Area")
        dockwidget.setFeatures(QDockWidget.DockWidgetMovable)

        font = QFont()
        font.setPointSize(16)
        self.m_logWidget = QPlainTextEdit()
        self.m_logWidget.setFont(font)
        dockwidget.setWidget(self.m_logWidget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dockwidget)

        MainLayout.setStretch(0, 2)
        MainLayout.setStretch(1, 2)
        MainLayout.setStretch(2, 1)


    def InitManager(self):
        self.Mgr = E_Manager(self)


        self.volumeWidget.SetManager(self.Mgr)



    def onImportObject(self):
        self.Mgr.SetLog('Import 3d Object')

        path = QFileDialog.getOpenFileName(self, "Import 3D Objects", "~/", "Object Files(*.stl *.obj) ;; Object Files(*.stl) ;; Object Files(*.obj)")
        self.Mgr.ImportObject(path[0])

    def onImportVolume(self):
        self.Mgr.SetLog('import Volume')

        path = QFileDialog.getOpenFileNames(self, "Import 3D Objects", "~/", "Dicom File(*.dcm)")
        fileSeries = path[0]

        if len(fileSeries) == 0: return

        #Import Volume
        self.Mgr.VolumeMgr.ImportVolume(fileSeries)

    def onInitNetwork(self):
        self.Mgr.InitNetwork()
        print()

    def onRandomPred(self):
        self.Mgr.RandomPrediction()

    def onMeshViewState(self, state):
        if state == 2: #show
            self.m_vtkWidget[0].show()
        else:
            self.m_vtkWidget[0].hide()

    def onVolumeViewState(self, state):
        if state == 2: #show
            self.m_vtkWidget[1].show()
        else:
            self.m_vtkWidget[1].hide()
    def onSliceViewState(self, state):
        if state==2:
            self.m_sliceViewWidget.show()
        else:
            self.m_sliceViewWidget.hide()


    def onSliderRangeChanged(self, min, max):
        obj = self.sender()
        obj.setSliderPosition( int((min + max) / 2) )

    def onSliderValueChanged(self, value):
        obj = self.sender()
        idx = self.m_sliceSlider.index(obj)

        self.Mgr.VolumeMgr.ChangeSliceIdx(idx, obj.value())

    def onRangeSliderValueChanged(self, value):
        xPos = self.m_rangeSlider[0].value() / 1000
        yPos = self.m_rangeSlider[1].value() / 1000

        self.Mgr.VolumeMgr.UpdateVolumeDataCrop(xPos, yPos)
