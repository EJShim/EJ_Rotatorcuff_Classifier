
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class E_VolumeTreeWidget(QTreeWidget):
    def __init__(self, parent = None):
        super(QTreeWidget, self).__init__(parent)

        self.mainFrm = parent

        self.setSortingEnabled(True)
        self.setHeaderHidden(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)

        self.itemDoubleClicked.connect(self.dbcEvent)

    def updateTree(self, info):
    
        self.clear()
        parent = QTreeWidgetItem(self)
        parent.setText(0, info['name'])

        serieses = info['serieses']
        for series in serieses:
            itemName = 'series ' + series + ' : ' + serieses[series]['description']
            
            child = QTreeWidgetItem(parent)
            child.setText(0, itemName)

            description = serieses[series]['description'].lower()

            #Fat Supression
            if not description.find('fs') == -1 or not description.find('fat') == -1:                
                child.setForeground(0, QBrush(QColor('blue')))
                if not description.find('cor') == -1:
                    child.setBackground(0, QBrush(QColor('green')))                        

            elif description.find('t1') == -1 and description.find('t2')== -1:
                child.setBackground(0, QBrush(QColor('red')))
        self.expandAll()


    def dbcEvent(self, item, col):
        itemName = item.text(col)
        itemIdx = itemName.split()[1]

        self.mainFrm.Mgr.VolumeMgr.AddSelectedVolume(itemIdx)