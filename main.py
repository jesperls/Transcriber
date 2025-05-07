from PySide6.QtWidgets import QApplication
from overlay import Overlay
import sys

if __name__ == '__main__':
    app = QApplication([])
    o = Overlay()
    o.show()
    sys.exit(app.exec())
