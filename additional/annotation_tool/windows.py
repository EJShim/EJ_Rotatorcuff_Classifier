import sys,os
import atexit

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
import manager


file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(file_path, os.pardir)
class AnnotationWindow(QMainWindow):    

    def __init__(self, parent = None):
        super(AnnotationWindow, self).__init__(parent)
        self.one_view = False
        self.selected_idx = None

        self.central_layout = QHBoxLayout()        
        self.installEventFilter(self)
        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(self.central_layout)        

        self.init_toolbar()
        self.init_list()
        self.init_mainfrm()
        self.set_view_grid()
        

        self.central_layout.setStretch(0, 1)
        self.central_layout.setStretch(1, 5)

        timer = QTimer(self)        
        timer.timeout.connect(self.update_timer)
        timer.start(1000)

        atexit.register(self.on_save_labels)


    def init_toolbar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)


        #Add Timer
        self.timer_widget = QLabel()
        toolbar.addWidget(self.timer_widget)
        toolbar.addSeparator()


        #Add Controler
        label_control = QGroupBox("data label")
        toolbar.addWidget(label_control)        
        self.layout_label_control = QHBoxLayout()
        label_control.setLayout(self.layout_label_control)
        
        radio_none = QRadioButton("None ('Q' key)")
        radio_none.toggled.connect(self.on_none_label)        
        
        radio_partial = QRadioButton("Partial ('W' key)")
        radio_partial.toggled.connect(self.on_partial_label)
        
        radio_small = QRadioButton("Small ('E' key)")
        radio_small.toggled.connect(self.on_small_label)        
        
        radio_medium = QRadioButton("Medium ('R' key)")
        radio_medium.toggled.connect(self.on_medium_label)
        
        radio_large = QRadioButton("Large+Massive ('T' key)")
        radio_large.toggled.connect(self.on_large_label)        
        


        self.layout_label_control.addWidget(radio_none)
        self.layout_label_control.addWidget(radio_partial)
        self.layout_label_control.addWidget(radio_small)
        self.layout_label_control.addWidget(radio_medium)
        self.layout_label_control.addWidget(radio_large)
        self.layout_label_control.itemAt(0).widget().setChecked(True)
        

    def init_list(self):
        self.widget_list = QListWidget()
        self.widget_list.setAlternatingRowColors(True)
        self.widget_list.itemDoubleClicked.connect(self.on_list_dbclicked)
        self.central_layout.addWidget(self.widget_list)

        self.update_list()

    def update_timer(self):
        manager.elapsed_time += 1

        secs = manager.elapsed_time % 60
        mins = manager.elapsed_time // 60
        hrs = manager.elapsed_time // 3600
        message = "elapsed time " + str(hrs).zfill(2) + ":" + str(mins).zfill(2) + ":" + str(secs).zfill(2)
        self.timer_widget.setText(message)


        #Save Label every 2 minutes
        if manager.elapsed_time % 120 == 0:
            self.on_save_labels()


        
    
    def update_list(self):
        #Load Data Annotation                
        annot_data = manager.tmp_data

        self.widget_list.clear()
        
        for idx, data in enumerate(annot_data):            
            if data == None:
                self.widget_list.insertItem(idx, str("(undefined)"))                
            elif data == 0: #NONE_RCT
                self.widget_list.insertItem(idx, str("None"))
                self.widget_list.item(idx).setBackground(QBrush(QColor('green')))
            elif data == 1: #Partial_RCT
                self.widget_list.insertItem(idx, str("Partial"))
                self.widget_list.item(idx).setBackground(QBrush(QColor(100, 0, 0)))
            elif data == 2: #RCT_RCT
                self.widget_list.insertItem(idx, str("Small"))
                self.widget_list.item(idx).setBackground(QBrush(QColor(150, 0, 0)))
            elif data == 3: #RCT_RCT
                self.widget_list.insertItem(idx, str("Medium"))
                self.widget_list.item(idx).setBackground(QBrush(QColor(200, 0, 0)))                    
            elif data == 4: #RCT_RCT
                self.widget_list.insertItem(idx, str("Large"))
                self.widget_list.item(idx).setBackground(QBrush(QColor(255, 0, 0 )))
                
            


    def init_mainfrm(self):
        self.widget_renderers = E_MainRenderingWidget()
        self.central_layout.addWidget(self.widget_renderers)    

        
        #Add Renderers to the widget
        renderers = manager.get_renderers()
        self.ren_widget = []

        for idx, renderer in enumerate(renderers):
            widget = QVTKRenderWindowInteractor()
            widget.GetRenderWindow().AddRenderer(renderers[idx])
            widget.GetRenderWindow().Render()

            #Interactor
            if idx == 0:
                interactor = manager.E_InteractorStyle()
            else:
                interactor = manager.E_InteractorStyle2D(idx = idx-1)

            widget.GetRenderWindow().GetInteractor().SetInteractorStyle(interactor)

            self.ren_widget.append(widget)
            
                

        # renderers = manager.get_renderers(

        self.widget_renderers.AddMainRenderer(self.ren_widget[0])    
        for i in range(1, 4):
            self.widget_renderers.AddSliceRenderer(self.ren_widget[i])


    #SLOTS
    def on_save_labels(self):
        manager.save_tmp_data()
    
    def set_view_normal(self):
        self.widget_renderers.SetViewMainView()
        
    def set_view_grid(self):
        self.widget_renderers.SetViewGridView()

    def on_list_dbclicked(self, item):
        for i in range(2):
            self.layout_label_control.itemAt(i).widget().setChecked(False)


        idx = self.widget_list.row(item)
        self.selected_idx = idx
        volume_arr = manager.get_volume(idx)        
        manager.add_volume(volume_arr)

    def toggle_view_mode(self):        
        selected_renderer = self.ren_widget[manager.selected_renderer[0]]
        self.one_view = not self.one_view

        if self.one_view:
            self.widget_renderers.SetViewOneView(selected_renderer)
        else:
            self.widget_renderers.SetViewFourView()

    def on_none_label(self):
        if self.selected_idx == None:
            return        

        #Update data and list
        manager.tmp_data[self.selected_idx] = 0
        self.update_list()

    def on_partial_label(self):
        if self.selected_idx == None:
            return

        manager.tmp_data[self.selected_idx] = 1
        self.update_list()

    def on_small_label(self):
        if self.selected_idx == None:
            return

        manager.tmp_data[self.selected_idx] = 2
        self.update_list()

    def on_medium_label(self):
        if self.selected_idx == None:
            return

        manager.tmp_data[self.selected_idx] = 3
        self.update_list()

    def on_large_label(self):
        if self.selected_idx == None:
            return

        manager.tmp_data[self.selected_idx] = 4
        self.update_list()




    def eventFilter(self, obj, event):
        # print(event)
        if event.type() == QEvent.ShortcutOverride:
            
            if event.key() == Qt.Key_Space:          
                self.toggle_view_mode()
            elif event.key() == Qt.Key_Q:
                self.layout_label_control.itemAt(1).widget().setChecked(True)
                self.layout_label_control.itemAt(0).widget().setChecked(True)
            elif event.key() == Qt.Key_W:
                self.layout_label_control.itemAt(0).widget().setChecked(True)
                self.layout_label_control.itemAt(1).widget().setChecked(True)
            elif event.key() == Qt.Key_E:
                self.layout_label_control.itemAt(0).widget().setChecked(True)
                self.layout_label_control.itemAt(2).widget().setChecked(True)
            elif event.key() == Qt.Key_R:
                self.layout_label_control.itemAt(0).widget().setChecked(True)
                self.layout_label_control.itemAt(3).widget().setChecked(True)
            elif event.key() == Qt.Key_T:
                self.layout_label_control.itemAt(0).widget().setChecked(True)
                self.layout_label_control.itemAt(4).widget().setChecked(True)
                

            return True # means stop event propagation
        else:
            return QMainWindow.eventFilter(self, obj, event)





class E_MainRenderingWidget(QWidget):
    def __init__(self, parent = None):
        super(E_MainRenderingWidget, self).__init__(parent)        

        self.mainLayout = QHBoxLayout()
        self.setLayout(self.mainLayout)
        
        self.sliceView = []
        self.mainView = None
        self.selectedView = None


        mainRenderWidget = QWidget()
        self.mainLayout.addWidget(mainRenderWidget)
        self.mainRenderLayout = QVBoxLayout()
        mainRenderWidget.setLayout(self.mainRenderLayout)


        sliceRenderWidget = QWidget()
        self.mainLayout.addWidget(sliceRenderWidget)
        self.sliceRenderLayout = QVBoxLayout()
        sliceRenderWidget.setLayout(self.sliceRenderLayout)



        #Set Spacing and Margins of Layouts
        self.mainLayout.setSpacing(0)
        self.mainLayout.setContentsMargins(0,0,0,0)

        self.mainRenderLayout.setSpacing(1)
        self.mainRenderLayout.setContentsMargins(0,0,1,0)

        self.sliceRenderLayout.setSpacing(1)
        self.sliceRenderLayout.setContentsMargins(0, 0, 0, 0)



        self.mainLayout.setStretch(0, 3)
        self.mainLayout.setStretch(1, 1)
    

    def AddMainRenderer(self, rendererWidget):        
        self.mainRenderLayout.addWidget(rendererWidget)
        self.mainView = rendererWidget
        self.selectedView = rendererWidget
    
    def AddSliceRenderer(self, rendererWidget):
        self.sliceRenderLayout.addWidget(rendererWidget)  
        self.sliceView.append(rendererWidget)

    def SetViewMainView(self):
        self.sliceView[0].setParent(self.sliceRenderLayout.parentWidget())
        self.sliceRenderLayout.insertWidget(0,self.sliceView)
        self.mainLayout.setStretch(0, 3)
        self.mainLayout.setStretch(1, 1)
        

    def SetViewGridView(self):        
        self.sliceView[0].setParent(self.mainRenderLayout.parentWidget())        
        self.mainRenderLayout.insertWidget(0,self.sliceView[0])
        self.mainRenderLayout.insertWidget(1, self.mainView)
        self.mainLayout.setStretch(0, 1)
        self.mainLayout.setStretch(1, 1)

    def SetViewOneView(self, renderingWidget):        
        self.selectedView = renderingWidget
        
        #Remove Widget from Main Layout
        self.mainView.setParent(self.sliceRenderLayout.parentWidget())     
        for widget in self.sliceView:
            widget.setParent(self.sliceRenderLayout.parentWidget())
        self.sliceRenderLayout.parentWidget().hide()


        #Add Render Widget To Main View
        renderingWidget.setParent(self.mainRenderLayout.parentWidget())
        self.mainRenderLayout.insertWidget(0,renderingWidget)

    def SetViewFourView(self):        
        self.sliceView[0].setParent(self.mainRenderLayout.parentWidget())
        self.mainRenderLayout.insertWidget(0,self.sliceView[0])
        self.mainView.setParent(self.mainRenderLayout.parentWidget())
        self.mainRenderLayout.insertWidget(1,self.mainView)
        
        self.sliceView[1].setParent(self.sliceRenderLayout.parentWidget())
        self.sliceRenderLayout.insertWidget(0,self.sliceView[1])
        self.sliceView[2].setParent(self.sliceRenderLayout.parentWidget())
        self.sliceRenderLayout.insertWidget(1,self.sliceView[2])

        self.sliceRenderLayout.parentWidget().show()
        

