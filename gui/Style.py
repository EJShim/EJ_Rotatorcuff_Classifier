styleData="""

QWidget
{
    color: #b1b1b1;
    background-color: #323232;
}
QListView {    
    alternate-background-color: #434343;
}

QTreeView { 
    alternate-background-color: #434343;
}
QProgressBar
{
    border: 2px solid grey;
    border-radius: 5px;
    text-align: center;
    color:#c1c1c1;
}
QProgressBar::chunk
{
    background-color: #1a80d7;
    width: 2.15px;
    margin: 0.5px;
}

QTabWidget::tab-bar {
    left: 5px;
}

QTabBar::tab {
    border-top-left-radius: 1px;
    border-top-right-radius: 1px;
    min-width: 8ex;
    padding: 1px;
}

QTabBar::tab:top:selected
{
    color:#EEEEEE;
    background-color: #1a80d7;
}

QTabBar::tab:top:!selected:hover {
    background-color: #2a90e7;
}

QHeaderView::section
{
    background-color: #76797C;
    color: #eff0f1;
    padding: 5px;
    border: 1px solid #FF0000;
}

QToolBar {
    border: 0px transparent;
}
"""