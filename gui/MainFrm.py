from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from gui.VolumeRenderingWidget import E_VolumeRenderingWidget
from gui.VolumeListWidget import E_VolumeListWidget
from gui.VolumeTreeWidget import E_VolumeTreeWidget
from gui.RendererViewWidget import E_MainRenderingWidget


import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import sys, os
from manager.Mgr import E_Manager
from manager.E_Threads import *


import numpy as np

file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
icon_path = os.path.join(root_path, "icons")

class E_MainWindow(QMainWindow):
    update_cam = pyqtSignal()


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


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.UpdateRenderer)

        #Random Prediction Animation Thread
        self.th_randomPred = ListAnimationThread(self)
        self.th_randomPred.predRandom.connect(self.onRandomPred)


        #Cam History Animation thread        
        self.th_camHistory = CamHistoryThread(self)        



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
        objectToolbar = QToolBar();
        objectToolbar.setIconSize(QSize(50, 50))        
        objectToolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        objectToolbar.setMovable(False)
        self.addToolBar(Qt.RightToolBarArea, objectToolbar)
        # mainTab.addTab(objectToolbar, "3D Objects")

        #Import Object Action
        importAction = QAction(QIcon(icon_path + "/051-cmyk.png"), "Import Object", self)
        importAction.triggered.connect(self.onImportObject)
        objectToolbar.addAction(importAction)

        #Import Volume addAction
        volumeAction = QAction(QIcon(icon_path + "/051-document.png"), "Import Volume", self)
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
        SaveAction = QAction(QIcon(icon_path + "/051-business-card.png"), "Save Processed Data", self)
        SaveAction.triggered.connect(self.onSaveData)
        objectToolbar.addAction(SaveAction)

        networkToolbar = QToolBar();
        networkToolbar.setIconSize(QSize(50, 50))
        networkToolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        networkToolbar.setMovable(False)
        self.addToolBar(networkToolbar)
        # mainTab.addTab(networkToolbar, "VRN")

        self.trainAction = QAction(QIcon(icon_path + "/051-pantone-2.png"), "Predict Off", self)
        self.trainAction.setCheckable(True)
        self.trainAction.toggled.connect(self.TogglePrediction)
        networkToolbar.addAction(self.trainAction)

        networkToolbar.addSeparator()

        self.listAnimation = QAction(QIcon(icon_path + "/051-pantone-1.png"), "List Animation", self)
        self.listAnimation.setCheckable(True)
        self.listAnimation.triggered.connect(self.onListAnimation)
        networkToolbar.addAction(self.listAnimation)

        self.camAnimation = QAction(QIcon(icon_path + "/051-cmyk.png"), "CAM Animation", self)        
        self.camAnimation.triggered.connect(self.onCAMAnimation)
        networkToolbar.addAction(self.camAnimation)


        # self.roiAnimation = QAction(QIcon(icon_path + "/051-cmyk.png"), "Predict ROI(experimental)", self)        
        # self.roiAnimation.triggered.connect(self.PredictROI)
        # networkToolbar.addAction(self.roiAnimation)


        # self.saveSlice = QAction(QIcon(icon_path + "/051-cmyk.png"), "Save Slice", self)        
        # self.saveSlice.triggered.connect(self.onSaveSliceImage)
        # networkToolbar.addAction(self.saveSlice)

        #Add Score Progress bar        
        style2 = "QProgressBar::chunk {background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0,stop: 0 #F10350,stop: 0.4999 #FF3320,stop: 0.5 #FF0019,stop: 1 #F0F150 );}"
        style3 = "QProgressBar::chunk {background: QLinearGradient( x1: 0, y1: 0, x2: 1, y2: 0,stop: 0 #F10350,stop: 0.4999 #FF3320,stop: 0.5 #FF0019,stop: 1 #05F150 );}"
        nonrctpro = QProgressBar()
        nonrctpro.setRange(0, 10000)
        nonrctpro.setValue(10000)        
        nonrctpro.setStyleSheet(style2)
        rctpro = QProgressBar()
        rctpro.setRange(0, 10000)
        rctpro.setValue(10000)
        rctpro.setStyleSheet(style3)
        self.score_group = QVBoxLayout()
        self.rctGroup.setSpacing(0)        
        self.score_group.addWidget(nonrctpro)
        self.score_group.addWidget(rctpro)
        groupbox_score = QGroupBox()
        groupbox_score.setLayout(self.score_group)
        networkToolbar.addWidget(groupbox_score)


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
        MainLayout.setStretch(2, 5)   

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

    def TogglePrediction(self, pred):        
        self.Mgr.m_bPred = pred
        
        if pred:
            self.trainAction.setText("Predict On")
        else:
            self.trainAction.setText("Predict Off")
        # self.Mgr.InitNetwork()
        # self.trainAction.setEnabled(False)
        # self.listAnimation.setEnabled(False)


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
        print("WTF")
        # if state==2:
        #     self.m_sliceViewWidget.show()
        # else:
        #     self.m_sliceViewWidget.hide()

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
            self.timer.start(30/1000)
            self.th_randomPred.start()
        else:
            self.th_randomPred.terminate()
            self.timer.stop()


    def onCAMAnimation(self, e):
        self.Mgr.RenderPreProcessedObject(self.th_camHistory.selectedIdx, predict=False)
        
        
        for idx, data in enumerate(self.th_camHistory.cam_history_data):
            self.Mgr.RotateCamera()
            self.Mgr.VolumeMgr.RemoveClassActivationMap()
            self.Mgr.VolumeMgr.AddClassActivationMap(data)        
            self.Mgr.Redraw()
            self.Mgr.Redraw2D()

            progress = int((idx/(self.th_camHistory.total_memory-1)) * 100.0)

            QApplication.processEvents()
            self.progressBar.setValue(progress)

    def onProgress(self, progress):
        self.progressBar.setValue(progress)

    def onMessage(self, msg):
        self.statusBar().showMessage(msg)

    def UpdateRenderer(self):        
        self.Mgr.RotateCamera()
        self.Mgr.Redraw()
        self.Mgr.Redraw2D()


    def PredictROI(self):
        self.Mgr.PredictROI()


    def onSaveSliceImage(self):
        self.Mgr.SaveSliceImage()

    def SetProgressScore(self, score, label=-1):

        msg = [
            "None RCT " + '{:.2f}'.format(score[0]*100.0) + "%",
            "RCT " + '{:.2f}'.format(score[1]*100.0) + "%"
        ]

        


        if not label == -1:

            pred_class = np.argmax(score)
            if int(label) == pred_class:
                msg[label] = "(correct) " + msg[label]
            else:
                msg[label] = "(wrong) " + msg[label]

            not_idx = not label
            self.score_group.itemAt(label).widget().setStyleSheet("QProgressBar::chunk{ background-color: green; }")
            self.score_group.itemAt(not_idx).widget().setStyleSheet("QProgressBar::chunk{ background-color: red; }")
        else:
            self.score_group.itemAt(0).widget().setStyleSheet("QProgressBar::chunk{ background-color: #1a80d7; }")
            self.score_group.itemAt(1).widget().setStyleSheet("QProgressBar::chunk{ background-color: #1a80d7; }")

        self.score_group.itemAt(0).widget().setValue(score[0] * 10000.0)
        self.score_group.itemAt(0).widget().setFormat(msg[0])
        self.score_group.itemAt(1).widget().setValue(score[1] * 10000.0)
        self.score_group.itemAt(1).widget().setFormat(msg[1])
