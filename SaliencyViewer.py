import logging
from PyQt5 import QtWidgets
from viewer import SaliencyWindow


def main():

    app = QtWidgets.QApplication([])
    window = SaliencyWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':

    logging.basicConfig(format='%(message)s', level=logging.INFO)

    main()
