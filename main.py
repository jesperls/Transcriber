from PySide6.QtWidgets import QApplication
from overlay import Overlay

if __name__ == '__main__':
    app = QApplication([])
    o = Overlay()
    o.show()
    app.exec()
