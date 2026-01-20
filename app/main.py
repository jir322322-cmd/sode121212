from __future__ import annotations

import sys

from PySide6 import QtWidgets

from app.gui import MapRestorerWindow


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MapRestorerWindow()
    window.resize(1200, 900)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
