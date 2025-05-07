import sys
import logging
from PySide6.QtWidgets import QApplication
from overlay import Overlay

def main():
    """Entry point for the application."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    app = QApplication(sys.argv)
    overlay = Overlay()
    overlay.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
