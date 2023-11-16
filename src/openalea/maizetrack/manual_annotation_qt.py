import cv2
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QPoint, QSize
from PySide6.QtGui import QPixmap, QScreen, QAction, QIcon, QImage
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QLabel, QScrollArea, QMainWindow, QDockWidget, QStatusBar, \
    QWidget, QApplication, QMenuBar, QMenu, QGroupBox, QGridLayout, QFileDialog, QLineEdit

from openalea.maizetrack.display import PALETTE


def _image(annot, task, angle):
    metainfo = annot[task]['metainfo']
    image = annot[task]['images'][angle]
    leaves_info = annot[task]['leaves_info']
    leaves_pl = [polylines[angle] for polylines in annot[task]['leaves_pl']]
    return rgb_and_polylines(image, leaves_pl, leaves_info, metainfo)


class ImageViewer(QWidget):
    def __init__(self, img=None):
        # initialize widget
        super().__init__()
        self.imageWidget = QLabel()
        self.imageWidget.installEventFilter(self)
        self.imageWidget.setAlignment(Qt.AlignCenter)
        if img is not None:
            self.pixmap = QPixmap.fromImage(img)
            self.imageWidget.setPixmap(self.pixmap)

        # create scroll area
        self.scrollArea = MyScrollArea(self.imageWidget)

        # insert to layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.scrollArea)
        self.setLayout(self.layout)

    def changeImagePath(self, path):
        image = cv2.imread(path)
        qImg = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qImg)
        self.imageWidget.setPixmap(self.pixmap)

    def changeImage(self, image, resize=True):
        qImg = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qImg)
        self.imageWidget.setPixmap(self.pixmap)
        if resize:
            self.imageWidget.resize(self.imageWidget.sizeHint())



class MyScrollArea(QScrollArea):
    def __init__(self, image_widget):
        # initialize widget
        super().__init__()
        self.setWidget(image_widget)
        self.myImageWidget = image_widget
        self.oldScale = 1
        self.newScale = 1
        self._zoom = 0
        self.mousepos = QPoint(0, 0)
        image_widget.setScaledContents(True)

    def wheelEvent(self, event) -> None:
        if event.angleDelta().y() < 0:
            # zoom out
            self.newScale = 0.8
        else:
            # zoom in
            self.newScale = 1.25

        widgetPos = self.myImageWidget.mapFrom(self, event.position())

        # resize image
        self.myImageWidget.resize(self.myImageWidget.size() * self.newScale)

        delta = widgetPos * self.newScale - widgetPos
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() + delta.x())
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().value() + delta.y())

        self.oldScale = self.newScale

    def mousePressEvent(self, event):
        self.mousepos = event.localPos()
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setCursor(Qt.OpenHandCursor)


    def mouseMoveEvent(self, event):
        delta = event.localPos() - self.mousepos
        # panning area
        if event.buttons() == Qt.MouseButton.MiddleButton:
            h = self.horizontalScrollBar().value()
            v = self.verticalScrollBar().value()

            self.horizontalScrollBar().setValue(int(h - delta.x()))
            self.verticalScrollBar().setValue(int(v - delta.y()))

        if event.buttons() == Qt.MouseButton.LeftButton:
            print("left click on image")
            par = self.parent().parent()
            pos = event.localPos()
            y, x = pos.y(), pos.x()

            dists = []
            polylines = [pl[par.angles[par.i_angle]] for pl in par.annot[par.tasks[par.i_task]]['leaves_pl']]
            for pl in polylines:
                d = min([np.linalg.norm(np.array([x, y]) - xy) for xy in pl])
                dists.append(d)
            i_selected = np.argmin(dists)
            selected_leaf = par.annot[par.tasks[par.i_task]]['leaves_info'][i_selected]
            selected_leaf['selected'] = True
            par.changeImage(_image(par.annot, par.tasks[par.i_task], par.angles[par.i_angle]), False)

        self.mousepos = event.localPos()

    def mouseReleaseEvent(self, event):
        self.unsetCursor()
        self.mousepos = event.localPos()


class DockAnnotation(QDockWidget):
    def __init__(self):
        super().__init__()
        # Setting up layout
        self.content = QWidget()
        self.contentLayout = QVBoxLayout(self.content)
        self.content.setLayout(self.contentLayout)
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)

        # adding stuff in
        groupBox = QGroupBox("Validation", self)

        self.contentLayout.addWidget(groupBox)
        self.paramsLayout = QGridLayout(groupBox)
        self.backButton = QPushButton('Back', self)
        self.forwardButton = QPushButton('Forward', self)
        self.minusOneButton = QPushButton('-1', self)
        self.plusOneButton = QPushButton('+1', self)
        self.okButton = QPushButton('Ok', self)
        self.cameraButton = QPushButton('Change Camera', self)
        self.rZero = QPushButton('r=0', self)
        self.rTen = QPushButton('r=10', self)
        self.plusOneAllButton = QPushButton('+1 all', self)
        self.minusOneAllButton = QPushButton('-1 all', self)
        groupBoxSelection = QGroupBox("Selection", self)

        groupBox.setMaximumSize(QSize(12656161, 500))
        groupBoxSelection.setMaximumSize(QSize(12656161, 200))

        self.contentLayout.addWidget(groupBoxSelection)
        self.selectionLayout = QGridLayout(groupBoxSelection)
        self.selectedLabel = QLabel("Selected:", self)
        self.selectedLineEdit = QLineEdit(self)

        self.setupButtonSignals()
        self.addWidgets()
        self.setWidget(self.content)

    def setupButtonSignals(self):
        self.cameraButton.clicked.connect(lambda: self.switch_cam())
        self.forwardButton.clicked.connect(lambda: self.changeTime(forward=True))
        self.backButton.clicked.connect(lambda: self.changeTime(forward=False))
        self.okButton.clicked.connect(lambda: self.validate())

    def addWidgets(self):
        self.paramsLayout.addWidget(self.backButton, 0, 1, 1, 1)
        self.paramsLayout.addWidget(self.forwardButton, 0, 2, 1, 1)
        self.paramsLayout.addWidget(self.minusOneButton, 0, 1, 2, 1)
        self.paramsLayout.addWidget(self.plusOneButton, 0, 2, 2, 1)
        self.paramsLayout.addWidget(self.rZero, 0, 1, 3, 1)
        self.paramsLayout.addWidget(self.rTen, 0, 2, 3, 1)
        self.paramsLayout.addWidget(self.minusOneAllButton, 0, 1, 4, 1)
        self.paramsLayout.addWidget(self.plusOneAllButton, 0, 2, 4, 1)
        self.paramsLayout.addWidget(self.okButton, 0, 1, 6, 2)
        self.paramsLayout.addWidget(self.cameraButton, 0, 1, 7, 2)

        self.selectionLayout.addWidget(self.selectedLabel, 0, 1, 1, 1)
        self.selectionLayout.addWidget(self.selectedLineEdit, 0, 2, 1, 1)

    def switch_cam(self):
        parent = self.parent()
        parent.i_angle = parent.i_angle + 1 if parent.i_angle < len(parent.angles) - 1 else 0
        parent.changeImage(_image(parent.annot, parent.tasks[parent.i_task], parent.angles[parent.i_angle]), False)

    def changeTime(self, forward=True):
        parent = self.parent()
        change = False
        if forward and parent.i_task < len(parent.tasks) - 1:
            change = True
            parent.i_task += 1
        elif not forward and parent.i_task > 0:
            change = True
            parent.i_task -= 1
        if change:
            parent.changeImage(_image(parent.annot, parent.tasks[parent.i_task], parent.angles[parent.i_angle]), False)

    def validate(self):
        parent = self.parent()
        selected_leaf = parent.annot[parent.tasks[parent.i_task]]['leaves_info'][parent.i_selected]
        selected_leaf['selected'] = False
        parent.changeImage(_image(parent.annot, parent.tasks[parent.i_task], parent.angles[parent.i_angle]), False)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.annot = {}
        self.angles = []
        self.tasks = []
        self.i_task = 0
        self.i_angle = 0
        self.img_dim = 0
        self.i_selected = -1
        self.dockWidget = DockAnnotation()
        self.setWindowTitle('Annotation Tool')
        self.imgViewer = ImageViewer()
        self.statusbar = QStatusBar(self)
        self.statusbar.showMessage("Starting application...")
        self.menubar = QMenuBar(self)
        self.addWidgets()
        self.resize(1280, 720)
        self.center()

    def changeImageDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open image file", "",
                                                  "png images (*.png);;All Files (*)")
        if fileName:
            self.imgViewer.changeImagePath(fileName)

    def changeImage(self, image, resize=True):
        self.imgViewer.changeImage(image, resize)

    def addWidgets(self):
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockWidget)
        self.setCentralWidget(self.imgViewer)
        self.setStatusBar(self.statusbar)
        helpAction = QAction("Help", self)
        loadImageAction = QAction("Load image", self)
        aboutAction = QAction(QIcon.fromTheme("help-about"), "About", self)
        quitAction = QAction(QIcon.fromTheme("application-exit"), "Quit", self)
        quitAction.triggered.connect(self.quit)
        helpAction.triggered.connect(self.help)
        loadImageAction.triggered.connect(self.changeImageDialog)
        menuFile = QMenu("File", self)
        menuHelp = QMenu("About", self)
        menuHelp.addAction(helpAction)
        menuHelp.addAction(aboutAction)
        menuFile.addAction(loadImageAction)
        menuFile.addAction(quitAction)

        self.menubar.addAction(menuFile.menuAction())
        self.menubar.addAction(menuFile.menuAction())
        self.menubar.addAction(menuHelp.menuAction())
        self.setMenuBar(self.menubar)

    def center(self):
        SrcSize = QScreen.availableGeometry(QApplication.primaryScreen())
        frmX = (SrcSize.width() - self.width()) / 2
        frmY = (SrcSize.height() - self.height()) / 2
        self.move(frmX, frmY)

    @staticmethod
    def help():
        print("help")

    def quit(self):
        self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.quit()

    def set_annotation_params(self, img_dimension, tasks, angles, i_task, i_angle):
        self.img_dim = img_dimension
        self.tasks = tasks
        self.angles = angles
        self.i_task = i_task
        self.i_angle = i_angle

    def annotate(self, annot):
        self.annot = annot
        tasks = list(annot.keys())
        angles = list(annot[tasks[0]]['images'].keys())
        i_task, i_angle = 0, 0

        img_dimension = annot[tasks[0]]['images'][angles[0]].shape[:2]

        self.set_annotation_params(img_dimension, tasks, angles, i_task, i_angle)
        self.changeImage(_image(annot, tasks[i_task], angles[i_angle]))
        self.statusbar.showMessage("Done!")


def rgb_and_polylines(image, leaves_pl, leaves_info, metainfo):
    image_pl = image.copy()

    for pl, leaf in zip(leaves_pl, leaves_info):

        # col = [0, 0, 0] if c == -2 else [int(x) for x in PALETTE[c]]
        col = [int(x) for x in PALETTE[leaf['rank'] - 1]]
        border_col = (0, 0, 0) if leaf['rank'] == 0 else (255, 255, 255)

        ds = 1 if not leaf['selected'] else 3

        image_pl = cv2.polylines(np.float32(image_pl),
                                 [pl.astype(int).reshape((-1, 1, 2))], False, border_col, 10 * ds)

        image_pl = cv2.polylines(np.float32(image_pl),
                                 [pl.astype(int).reshape((-1, 1, 2))], False, col, 7 * ds)

        # tip if mature
        if leaf['mature']:
            pos = (int(pl[-1][0]), int(pl[-1][1]))
            image_pl = cv2.circle(np.float32(image_pl), pos, 20, (0, 0, 0), -1)

        # rank number
        pos = (int(pl[-1][0]), int(pl[-1][1]))
        image_pl = cv2.putText(image_pl, str(leaf['rank']), pos, cv2.FONT_HERSHEY_SIMPLEX,
                               3, (0, 0, 0), 4, cv2.LINE_AA)

    # write date
    if metainfo is not None:
        text = 'plantid {} / task {} ({})'.format(metainfo.pot, metainfo.task, metainfo.daydate)
        cv2.putText(image_pl, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 10, cv2.LINE_AA)
    image_pl = image_pl.astype(np.uint8)
    return image_pl
