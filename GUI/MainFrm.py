from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import sys, os
from Manager.Mgr import E_Manager
from GUI.VolumeRenderingWidget import E_VolumeRenderingWidget
from GUI.VolumeListWidget import E_VolumeListWidget
from GUI.VolumeTreeWidget import E_VolumeTreeWidget

import numpy as np

curPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.abspath(os.path.join(curPath, os.pardir))
iconPath = rootPath + "/icons"

class E_MainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(E_MainWindow, self).__init__(parent)

        self.m_saveDir = None;

        self.setWindowTitle("EJ ModelNet Project")
        self.keyPlaying = {}


        #Central Widget
        self.m_centralWidget = QWidget()
        self.setCentralWidget(self.m_centralWidget)


        #Volume List Dialog
        self.m_volumeListDlg = E_VolumeListWidget()

        #vtk Renderer Widget
        self.m_vtkWidget = [0, 0]
        for i in range(2):
            self.m_vtkWidget[i] = QVTKRenderWindowInteractor();

        #Bone Color, RCT
        self.m_bBoneColorBlack = "Black"
        self.m_bRCT = True


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

        #Show/hide Volume List
        treeViewCheck = QCheckBox("Volume Tree")
        treeViewCheck.setCheckState(2)
        treeViewCheck.stateChanged.connect(self.onVolumeTreeState)
        checkLayout.addWidget(treeViewCheck)


        listViewCheck = QCheckBox("List View")
        listViewCheck.setCheckState(0)
        listViewCheck.stateChanged.connect(self.onListViewState)
        checkLayout.addWidget(listViewCheck)

        meshViewCheck = QCheckBox("Mesh View")
        meshViewCheck.setCheckState(0)
        meshViewCheck.stateChanged.connect(self.onMeshViewState)
        checkLayout.addWidget(meshViewCheck)

        volumeViewCheck = QCheckBox("Volume View")
        volumeViewCheck.setCheckState(2)
        volumeViewCheck.stateChanged.connect(self.onVolumeViewState)
        checkLayout.addWidget(volumeViewCheck)


        sliceViewCheck = QCheckBox("Slice View")
        sliceViewCheck.setCheckState(2)
        sliceViewCheck.stateChanged.connect(self.onSliceViewState)
        checkLayout.addWidget(sliceViewCheck)


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

        objectToolbar.addSeparator()

        self.radioGroubBox = QGroupBox("Bone Color")
        bone_black = QRadioButton("Black")
        bone_white = QRadioButton("White")
        bone_black.setChecked(True);

        grouopBoxLayout = QVBoxLayout()
        grouopBoxLayout.addWidget(bone_black)
        grouopBoxLayout.addWidget(bone_white)

        self.radioGroubBox.setLayout(grouopBoxLayout)

        bone_black.toggled.connect(self.onChangeTF)
        objectToolbar.addWidget(self.radioGroubBox)

        objectToolbar.addSeparator()


        self.RCTGroupBox = QGroupBox("RCT")

        self.radio_RCT = QRadioButton("RCT")
        self.radio_NRCT = QRadioButton("Non-RCT")
        self.radio_RCT.setChecked(True);
        self.radio_RCT.toggled.connect(self.onChangeRCT)

        grouopBoxLayout2 = QVBoxLayout()
        grouopBoxLayout2.addWidget(self.radio_RCT)
        grouopBoxLayout2.addWidget(self.radio_NRCT)

        self.RCTGroupBox.setLayout(grouopBoxLayout2)

        objectToolbar.addWidget(self.RCTGroupBox)

        objectToolbar.addSeparator()


        #Save Object Action
        SaveAction = QAction(QIcon(iconPath + "/051-cmyk.png"), "Save Processed Data", self)
        SaveAction.triggered.connect(self.onSaveData)
        objectToolbar.addAction(SaveAction)



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
        self.m_sliceViewWidget


    def InitCentralWidget(self):
        MainLayout = QHBoxLayout()
        self.m_centralWidget.setLayout(MainLayout)

        self.m_listWidget = E_VolumeListWidget(self)        
        self.m_listWidget.hide()
        MainLayout.addWidget(self.m_listWidget)

        self.m_treeWidget = E_VolumeTreeWidget(self)
        MainLayout.addWidget(self.m_treeWidget)


        for i in range(2):
            self.m_vtkWidget[i]
            MainLayout.addWidget(self.m_vtkWidget[i])

        self.m_vtkWidget[0].hide()
        MainLayout.addWidget(self.m_sliceViewWidget)

        #dock widget
        dockwidget = QDockWidget("LOG")
        dockwidget.setFeatures(QDockWidget.DockWidgetMovable)

        font = QFont()
        font.setPointSize(16)
        self.m_logWidget = QPlainTextEdit()
        self.m_logWidget.setFont(font)
        dockwidget.setWidget(self.m_logWidget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dockwidget)

        MainLayout.setStretch(0, 1)
        MainLayout.setStretch(1, 1)
        MainLayout.setStretch(2, 2)
        MainLayout.setStretch(3, 2)
        MainLayout.setStretch(4, 0.5)


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

        dirName = os.path.dirname(str(path[0][0]))
        dirName = str(dirName).lower()

        if not dirName.find('none') == -1 and not dirName.find('rct') == -1:
            self.Mgr.SetLog("None-RCT Data")
            self.radio_NRCT.setChecked(True)
        else:
            self.Mgr.SetLog("RCT Data")
            self.radio_RCT.setChecked(True)        

        if len(fileSeries) == 0: return


        #Import Volume        
        try :
            self.Mgr.VolumeMgr.ImportVolume(fileSeries)
        except Exception as e:
            self.Mgr.SetLog(str(e))

        self.Mgr.Redraw()
        self.Mgr.Redraw2D()

    def onSaveData(self):
        
        if self.m_saveDir == None:
            self.Mgr.SetLog("No save data available")
        try:            
            self.Mgr.SetLog(self.m_saveDir)

            Series = self.Mgr.VolumeMgr.GetSelectedSeries()

            if Series == None:                
                return

            orientation = Series['orientation']
            protocol = Series['protocol']
            

            log = "Save Processed Data in" + self.m_saveDir + '/Save_' + orientation + "_" + protocol + ".npz"            
            self.Mgr.SetLog(log)




            # savePath = os.path.join(rootPath, "ManualData")
            # savePath = os.path.join(savePath, self.m_bBoneColorBlack)

            # fileName = str(self.m_bBoneColorBlack) + "_" + str(len(os.listdir(savePath)))
            # savePath = os.path.join(savePath, fileName)

            # log = "Save Processed Data : " +str(savePath)
            # self.Mgr.SetLog(log)

            # log = "RCT : " + str(int(self.m_bRCT))
            # self.Mgr.SetLog(log)



            # volume = self.Mgr.VolumeMgr.m_volumeData

            # for rot in range(4):
            #     volume = np.rot90(volume, rot)
            #     features.append(volume)
            #     targets.append(int(self.m_bRCT))
            #     colors.append(self.m_bBoneColorBlack)

            # #Save To the Path
            # X = np.asarray(features)
            # X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2], X.shape[3])
            # targets = np.asarray(targets)
            # colors = np.asarray(colors)

            # np.savez_compressed( savePath, features=X, targets=targets, colors=colors)
            # self.Mgr.SetLog("Save Done!")


        except Exception as e:
            e = "***Error Raised : " + str(e)
            self.Mgr.SetLog(e)

    def onInitNetwork(self):
        self.Mgr.InitNetwork()

    def onRandomPred(self):
        self.Mgr.RandomPrediction()

    def onListViewState(self, state):
        if state == 2:
            self.m_listWidget.show()
        else:
            self.m_listWidget.hide()

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

    def onVolumeTreeState(self, state):
        if state == 2: #show
            self.m_treeWidget.show()
        else:
            self.m_treeWidget.hide()
        

    def onChangeRCT(self, isTrue):
        self.m_bRCT = isTrue

    def onChangeTF(self, isTrue):

        if isTrue:
            self.m_bBoneColorBlack = "Black"
        else:
            self.m_bBoneColorBlack = "White"


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
        self.Mgr.Redraw()
        self.Mgr.Redraw2D()

    def keyPressEvent(self, e):

        if e.key() == 65:
            self.m_rangeSlider[0].setValue(self.m_rangeSlider[0].value()-1)
        elif e.key() == 83:
            self.m_rangeSlider[0].setValue(self.m_rangeSlider[0].value()+1)
        elif e.key() == 90:
            self.m_rangeSlider[1].setValue(self.m_rangeSlider[0].value()-1)
        elif e.key() == 88:
            self.m_rangeSlider[1].setValue(self.m_rangeSlider[0].value()+1)
        else:
            return
