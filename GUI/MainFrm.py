#RADIO BUTTON -> DropDown Menu = not now

#Renderer View  - FInisehd
#TESTDATA Animation Show - Finisehd


#CLASS Activation Map Animation Show -- Use Test Data ids no.4 or no.24
#Use Multi-Thread when compiling theano functions

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import sys, os
from Manager.Mgr import E_Manager
from Manager.E_Threads import *
from GUI.VolumeRenderingWidget import E_VolumeRenderingWidget
from GUI.VolumeListWidget import E_VolumeListWidget
from GUI.VolumeTreeWidget import E_VolumeTreeWidget
from GUI.RendererViewWidget import E_MainRenderingWidget


import numpy as np

curPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.abspath(os.path.join(curPath, os.pardir))
iconPath = rootPath + "/icons"

class E_MainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(E_MainWindow, self).__init__(parent)

        self.m_saveDir = None;

        self.setWindowTitle("VRN Rotator-Cuff-Tear Classifier")
        self.keyPlaying = {}


        #Central Widget
        self.m_centralWidget = QWidget()
        self.setCentralWidget(self.m_centralWidget)


        #Volume List Dialog
        self.m_volumeListDlg = E_VolumeListWidget()

        #Random Prediction Animation Thread
        self.th_randomPred = ListAnimationThread(self)
        self.th_randomPred.predRandom.connect(self.onRandomPred)


        #Cam History Animation thread
        self.cam_history_data = None
        self.th_camHistory = CamHistoryThread(self)
        self.th_camHistory.cam_data.connect(self.updateCAM)
        self.th_camHistory.onprogress.connect(self.onProgress)
        self.th_camHistory.finished.connect(self.onFinishedCamHistory)

        

        

        #Bone Color, RCT
        self.m_bBoneColorBlack = "Black"
        self.m_bRCT = True        
        self.m_sliceSlider = [0, 0, 0]

        #vtk Renderer Widget
        self.m_vtkWidget = [0, 0]
        self.m_vtkSliceWidget = [0,0,0]      


        #Initialize

        self.InitToolbar()
        self.InitCentralWidget()
        # self.InitSliceViewWidget()
        self.InitManager()



        #Status Bar
        self.statusBar().showMessage('Ready')
        self.progressBar = QProgressBar()
        self.progressBar.setGeometry(30, 40, 200, 25)        
        self.statusBar().addPermanentWidget(self.progressBar)


    def InitToolbar(self):
        #ToolBar
        toolbar = QToolBar()

        self.addToolBar(toolbar)


        mainTab = QTabWidget()
        toolbar.addWidget(mainTab)


        objectToolbar = QToolBar();
        objectToolbar.setIconSize(QSize(50, 50))        
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
        checkLayout.setSpacing(0)
        
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

        self.classCheck = QCheckBox("CAM")
        self.classCheck.setCheckState(2)
        self.classCheck.setEnabled(False)
        self.classCheck.stateChanged.connect(self.onClassActivationMapState)
        checkLayout.addWidget(self.classCheck)
    
        objectToolbar.addSeparator()

        ##View 1, 4 View
        viewcontrolLayout = QVBoxLayout()
        viewControl = QGroupBox("View Control")
        viewControl.setLayout(viewcontrolLayout)
        radioNormal = QRadioButton("Normal View")
        radioNormal.clicked.connect(self.SetViewModeNromal)
        radioGrid = QRadioButton("Grid View")
        radioGrid.clicked.connect(self.SetViewModeGrid)


        viewcontrolLayout.addWidget(radioNormal)
        viewcontrolLayout.addWidget(radioGrid)                   
        viewcontrolLayout.itemAt(0).widget().setChecked(True)        
        objectToolbar.addWidget(viewControl)

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


        #RCT Group
        self.rctGroup = QVBoxLayout()
        self.rctGroup.setSpacing(0)
        self.rctGroup.setContentsMargins(0, 0, 0, 0)        
        groupBoxRCT = QGroupBox("RCT")
        groupBoxRCT.setLayout(self.rctGroup)                        
        self.rctGroup.addWidget(QRadioButton("None"))
        self.rctGroup.addWidget(QRadioButton("Small"))
        self.rctGroup.addWidget(QRadioButton("Medium"))
        self.rctGroup.addWidget(QRadioButton("Large"))
        self.rctGroup.addWidget(QRadioButton("Massive"))
        self.rctGroup.itemAt(0).widget().setChecked(True)
        objectToolbar.addWidget(groupBoxRCT)
        objectToolbar.addSeparator()


        self.orientationGroup = QVBoxLayout()
        groupBoxOri = QGroupBox("Orientation")
        groupBoxOri.setLayout(self.orientationGroup)
        self.orientationGroup.addWidget(QRadioButton("AXL"))
        self.orientationGroup.addWidget(QRadioButton("COR"))
        self.orientationGroup.addWidget(QRadioButton("SAG"))                
        self.orientationGroup.itemAt(0).widget().setChecked(True)
        objectToolbar.addWidget(groupBoxOri)

        objectToolbar.addSeparator()        


        #Protocol Group
        protocolAndSeriesWidget = QWidget()
        objectToolbar.addWidget(protocolAndSeriesWidget)
        protocolAndSeriesWidget.setLayout(QVBoxLayout())
        protocolAndSeriesWidget.layout().setSpacing(0)


        self.protocolGroup = QHBoxLayout()
        groupBoxPro = QGroupBox("Protocol")
        groupBoxPro.setLayout(self.protocolGroup)
        protocolAndSeriesWidget.layout().addWidget(groupBoxPro)
        self.protocolGroup.addWidget(QRadioButton("T1"))
        self.protocolGroup.addWidget(QRadioButton("T2"))        
        self.protocolGroup.itemAt(0).widget().setChecked(True)


        SeriesWidget = QWidget()
        SeriesWidget.setLayout(QHBoxLayout())
        protocolAndSeriesWidget.layout().addWidget(SeriesWidget)

        SeriesWidget.layout().addWidget(QLabel("Series : "))        
        self.m_SeriesNumber = QLabel("unknown") 
        SeriesWidget.layout().addWidget(self.m_SeriesNumber)
        
        
        
        objectToolbar.addSeparator()        

        #Save Object Action
        SaveAction = QAction(QIcon(iconPath + "/051-business-card.png"), "Save Processed Data", self)
        SaveAction.triggered.connect(self.onSaveData)
        objectToolbar.addAction(SaveAction)



        networkToolbar = QToolBar();
        networkToolbar.setIconSize(QSize(50, 50))
        networkToolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        mainTab.addTab(networkToolbar, "VRN")

        self.trainAction = QAction(QIcon(iconPath + "/051-pantone-2.png"), "Initialize Network", self)
        self.trainAction.triggered.connect(self.onInitNetwork)
        networkToolbar.addAction(self.trainAction)

        predAction = QAction(QIcon(iconPath + "/051-programming.png"), "Predict Random", self)
        predAction.triggered.connect(self.onRandomPred)
        networkToolbar.addAction(predAction)

        networkToolbar.addSeparator()

        self.listAnimation = QAction(QIcon(iconPath + "/051-pantone-1.png"), "List Animation", self)
        self.listAnimation.setCheckable(True)
        self.listAnimation.triggered.connect(self.onListAnimation)
        networkToolbar.addAction(self.listAnimation)

        camAnimation = QAction(QIcon(iconPath + "/051-cmyk.png"), "CAM Animation", self)        
        camAnimation.triggered.connect(self.onCAMAnimation)
        networkToolbar.addAction(camAnimation)


        
        toolbar.setFixedHeight(140)


    def InitRendererView(self, layout):

        self.renderViewWidget = E_MainRenderingWidget()
        layout.addWidget(self.renderViewWidget)        


        #Initialize Renderers
        for i in range(2):
            self.m_vtkWidget[i] = QVTKRenderWindowInteractor();
            self.renderViewWidget.AddMainRenderer(self.m_vtkWidget[i])

        self.m_vtkWidget[0].hide()


        for i in range(3):            
            #Slice View            
            self.m_vtkSliceWidget[i] = QVTKRenderWindowInteractor();

            #Slice Image Slider
            self.m_sliceSlider[i] = QSlider(Qt.Horizontal)
            self.m_sliceSlider[i].setRange(0, 0)
            self.m_sliceSlider[i].setSingleStep(1)

            self.m_sliceSlider[i].rangeChanged.connect(self.onSliderRangeChanged)
            self.m_sliceSlider[i].valueChanged.connect(self.onSliderValueChanged)

            self.renderViewWidget.AddSliceRenderer(self.m_vtkSliceWidget[i])        



    def InitCentralWidget(self):
        MainLayout = QHBoxLayout()
        MainLayout.setSpacing(0)
        MainLayout.setContentsMargins(0,0,0,0)
        self.m_centralWidget.setLayout(MainLayout)

        self.m_listWidget = E_VolumeListWidget(self)        
        self.m_listWidget.hide()
        MainLayout.addWidget(self.m_listWidget)

        self.m_treeWidget = E_VolumeTreeWidget(self)
        MainLayout.addWidget(self.m_treeWidget)


        #Initialize Main View
        self.InitRendererView(MainLayout)                
        

        #dock widget
        dockwidget = QDockWidget()
        dockwidget.setMaximumHeight(100)
        # dockwidget.setFeatures(QDockWidget.DockWidgetMovable)
        font = QFont()
        font.setPointSize(16)
        self.m_logWidget = QPlainTextEdit()
        self.m_logWidget.setFont(font)
        dockwidget.setWidget(self.m_logWidget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dockwidget)

        MainLayout.setStretch(0, 1)
        MainLayout.setStretch(1, 1)
        MainLayout.setStretch(2, 4)          

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
        self.m_saveDir = dirName
        
        dirName = str(dirName).lower()

    
        if not dirName.find('none') == -1:
            self.Mgr.SetLog("None-RCT Data")
            self.rctGroup.itemAt(0).widget().setChecked(True)
        elif not dirName.find('small') == -1:
            self.Mgr.SetLog("Small RCT Data")
            self.rctGroup.itemAt(1).widget().setChecked(True)
        elif not dirName.find('medium') == -1:
            self.Mgr.SetLog("Medium RCT Data")
            self.rctGroup.itemAt(2).widget().setChecked(True)
        elif not dirName.find('large') == -1:
            self.Mgr.SetLog("Large RCT Data")
            self.rctGroup.itemAt(3).widget().setChecked(True)
        elif not dirName.find('massive') == -1:
            self.Mgr.SetLog("Massive RCT Data")
            self.rctGroup.itemAt(4).widget().setChecked(True)

        if len(fileSeries) == 0: return


        #Import Volume        
        try :
            self.Mgr.VolumeMgr.ImportVolume(fileSeries)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            
            self.Mgr.SetLog(str(exc_type), error=True)
            self.Mgr.SetLog(str(fname), error=True)
            self.Mgr.SetLog(str(exc_tb.tb_lineno), error=True)
            self.Mgr.SetLog(str(e), error=True)

        self.Mgr.Redraw()
        self.Mgr.Redraw2D()

    def onSaveData(self):
        
        if self.m_saveDir == None:
            self.Mgr.SetLog("No save data available", error=True)
            return
        if self.Mgr.VolumeMgr.m_resampledVolumeData.any() == None:
            self.Mgr.SetLog("Volume Data Need to be Resampled", error=True)
            return            
        try:
            orientation = 'unknown'
            for idx in range(0, self.orientationGroup.count()):
                item = self.orientationGroup.itemAt(idx).widget()
                if item.isChecked():
                    orientation = item.text()
                    break

            protocol = 'unknown'
            for idx in range(0, self.protocolGroup.count()):
                item = self.protocolGroup.itemAt(idx).widget()
                if item.isChecked():
                    protocol = item.text()
                    break

            rct = 'unknown'
            for idx in range(0, self.rctGroup.count()):
                item = self.rctGroup.itemAt(idx).widget()
                if item.isChecked():
                    rct = item.text()
                    break

            

            #Save Series and Resampling Position For Futer Work
            series = int(self.m_SeriesNumber.text())
            xPos = self.m_rangeSlider[0].value() / 1000
            yPos = self.m_rangeSlider[1].value() / 1000



            savePath = self.m_saveDir + '/' + rct + "_" + orientation + "_" + protocol
            log = "Save Processed Data in (" + savePath   + ".npz" +  ")"            
            self.Mgr.SetLog(log)

            # saveData = dict(series = series, 
            #                 x = xPos, y = yPos, 
            #                 status = rct, 
            #                 orientation = orientation, 
            #                 protocol = protocol, 
            #                 data=self.Mgr.VolumeMgr.m_resampledVolumeData)
            
            np.savez_compressed(savePath, series = series, 
                                            x = xPos, y = yPos, 
                                            status = rct, 
                                            orientation = orientation, 
                                            protocol = protocol, 
                                            data=self.Mgr.VolumeMgr.m_resampledVolumeData)


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]


            self.Mgr.SetLog(exc_type, error=True)
            self.Mgr.SetLog(fname, error=True)
            self.Mgr.SetLog(exc_tb.tb_lineno, error=True)
            self.Mgr.SetLog(str(e), error=True)

    def onInitNetwork(self):
        self.Mgr.InitNetwork()

        self.trainAction.setEnabled(False)
        self.listAnimation.setEnabled(False)

    def onRandomPred(self):        
        self.Mgr.RandomPrediction()
        self.Mgr.Redraw()

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


    def onSliceViewState(self, state):
        if state==2:
            self.m_sliceViewWidget.show()
        else:
            self.m_sliceViewWidget.hide()

    def onClassActivationMapState(self, state):
        self.Mgr.VolumeMgr.ToggleClassActivationMap(state)


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
    


    def SetViewModeNromal(self):
        self.renderViewWidget.SetViewMainView()

    def SetViewModeGrid(self):
        self.renderViewWidget.SetViewGridView()


    def onListAnimation(self, e):
        if e:
            self.th_randomPred.start()
        else:
            self.th_randomPred.terminate()


    def onCAMAnimation(self, e):
        self.th_camHistory.start()
        self.Mgr.RenderPreProcessedObject(self.th_camHistory.selectedIdx)

        self.statusBar().showMessage('Class Animation History among epochs')

    def updateCAM(self, array):
        self.th_camHistory.updating = True
        self.Mgr.ClearScene()
        self.Mgr.RenderPreProcessedObject(self.th_camHistory.selectedIdx)

        
        # self.Mgr.ClearCAM()
        self.Mgr.VolumeMgr.AddClassActivationMap(array)
        self.th_camHistory.updating = False
        self.Mgr.Redraw()
        self.Mgr.Redraw2D()

    def onFinishedCamHistory(self):        
        self.statusBar().showMessage('Finished CAM')
        self.progressBar.setValue(0)

    def onProgress(self, progress):
        self.progressBar.setValue(progress)