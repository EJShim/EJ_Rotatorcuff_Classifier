import sys,os

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
import manager


file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(file_path, os.pardir)
icon_path = os.path.join(root_path, 'icons')

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

        


    def init_toolbar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        #Import Object Action
        save_action = QAction(QIcon(icon_path + "/051-cmyk.png"), "Save Label", self)
        save_action.triggered.connect(self.on_save_labels)
        toolbar.addAction(save_action)


        #View Control
         ##View 1, 4 View
        view_control = QGroupBox("View Control")
        toolbar.addWidget(view_control)
        layout_view_control = QVBoxLayout()
        view_control.setLayout(layout_view_control)
        radio_normal = QRadioButton("Normal View")
        radio_normal.clicked.connect(self.set_view_normal)
        radio_normal.setEnabled(False)
        radio_grid = QRadioButton("Grid View")
        radio_grid.clicked.connect(self.set_view_grid)
        layout_view_control.addWidget(radio_grid)
        layout_view_control.addWidget(radio_normal)
        layout_view_control.itemAt(0).widget().setChecked(True)



        label_control = QGroupBox("data label")
        toolbar.addWidget(label_control)        
        self.layout_label_control = QVBoxLayout()
        label_control.setLayout(self.layout_label_control)
        radio_none = QRadioButton("None ('X' key)")
        radio_none.toggled.connect(self.on_none_label)        
        radio_rct = QRadioButton("RCT ('Z' key)")
        radio_rct.toggled.connect(self.on_rct_label)
        self.layout_label_control.addWidget(radio_none)
        self.layout_label_control.addWidget(radio_rct)                   
        self.layout_label_control.itemAt(0).widget().setChecked(True)
        

    def init_list(self):
        self.widget_list = QListWidget()
        self.widget_list.setAlternatingRowColors(True)
        self.widget_list.itemDoubleClicked.connect(self.on_list_dbclicked)
        self.central_layout.addWidget(self.widget_list)

        self.update_list()


        
    
    def update_list(self):
        #Load Data Annotation                
        annot_data = manager.tmp_data

        self.widget_list.clear()
        
        for idx, data in enumerate(annot_data):            
            if data == None:
                self.widget_list.insertItem(idx, str("(undefined)"))                
            elif data == 0: #NONE_RCT
                self.widget_list.insertItem(idx, str("None-RCT"))
                self.widget_list.item(idx).setBackground(QBrush(QColor('green')))
            elif data == 1: #RCT_RCT
                self.widget_list.insertItem(idx, str("RCT"))
                self.widget_list.item(idx).setBackground(QBrush(QColor('red')))

            


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
        print("save labels")
    
    def set_view_normal(self):
        self.widget_renderers.SetViewMainView()
        
    def set_view_grid(self):
        self.widget_renderers.SetViewGridView()

    def on_list_dbclicked(self, item):
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

    def on_rct_label(self):
        if self.selected_idx == None:
            return

        manager.tmp_data[self.selected_idx] = 1
        self.update_list()




    def eventFilter(self, obj, event):
        # print(event)
        if event.type() == QEvent.ShortcutOverride:
            
            if event.key() == Qt.Key_Space:          
                self.toggle_view_mode()
            elif event.key() == Qt.Key_X:
                self.layout_label_control.itemAt(0).widget().setChecked(True)
            elif event.key() == Qt.Key_Z:
                self.layout_label_control.itemAt(1).widget().setChecked(True)
                

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
        print("change view")
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
        

