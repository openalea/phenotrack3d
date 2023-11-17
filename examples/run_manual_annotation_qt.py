import sys

from PySide6 import QtWidgets

from datadir import datadir
from openalea.maizetrack.manual_annotation_qt import Window

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.processEvents()
    w = Window()
    w.startup(datadir)
    w.show()

    sys.exit(app.exec())
