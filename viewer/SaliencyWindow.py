from PyQt5.QtWidgets import QWidget, QMainWindow, QDesktopWidget, \
            QGridLayout, QPushButton, QLabel
from PyQt5.QtGui import QFont

import pyqtgraph.opengl as gl
from . import common, model_inference

class SaliencyWindow(QMainWindow):

    def __init__(self) -> None:
        super(SaliencyWindow, self).__init__()

        self.monitor = QDesktopWidget().screenGeometry(0)
        self.monitor.setHeight(int(self.monitor.height() * 0.5))
        self.monitor.setWidth(int(self.monitor.width() * 0.5))

        self.grid_dimensions = 20
        self.index = 0

        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)
        self.layout = QGridLayout()

        self.viewer = common.AL_viewer()
        self.viewer2 = common.AL_viewer()

        self.grid = gl.GLGridItem()

        self.init_window()

    def init_window(self):
        self.centerWidget.setLayout(self.layout)

        self.viewer.setWindowTitle('drag & drop point cloud viewer')
        self.viewer.setCameraPosition(distance=2 * self.grid_dimensions)
        self.layout.addWidget(self.viewer, 0, 0, 1, 4)
        self.viewer2.setWindowTitle('drag & drop point cloud viewer')
        self.viewer2.setCameraPosition(distance=2 * self.grid_dimensions)
        self.layout.addWidget(self.viewer2, 0, 6, 1, 4)

        # grid
        # self.grid.setSize(self.grid_dimensions, self.grid_dimensions)
        # self.grid.setSpacing(1, 1)
        # self.grid.translate(0, 0, -2)
        # self.viewer.addItem(self.grid)
        # self.viewer2.addItem(self.grid)
        # File Label
        self.file_name_label = QLabel()
        self.layout.addWidget(self.file_name_label, 1, 1, 1, 3)

        # Buttons
        # Load data
        self.load_kitti_btn = QPushButton("KITTI")
        self.layout.addWidget(self.load_kitti_btn, 1, 0, 1, 1)
        self.load_kitti_btn.pressed.connect(self.load_kitti_instance)
        # < > select
        self.prev_btn = QPushButton("<-")
        self.layout.addWidget(self.prev_btn, 1, 6, 1, 2)
        self.prev_btn.clicked.connect(self.decrement_index)
        self.prev_btn.setEnabled(False)

        self.next_btn = QPushButton("->")
        self.layout.addWidget(self.next_btn, 1, 8, 1, 2)
        self.next_btn.clicked.connect(self.increment_index)
        self.next_btn.setEnabled(False)

        # Saliency Class
        self.saliency_class = model_inference.Salency_Class()

        # Predict
        self.inference_classific_btn = QPushButton("Inference")
        self.layout.addWidget(self.inference_classific_btn, 2, 0, 1, 1)
        self.inference_classific_btn.pressed.connect(self.predict)
        self.inference_classific_btn.setEnabled(False)

        # Saliency
        self.show_saliency_btn = QPushButton("Saliency")
        self.layout.addWidget(self.show_saliency_btn, 2, 1, 1, 1)
        self.show_saliency_btn.clicked.connect(self.show_saliency)
        self.show_saliency_btn.setEnabled(False)

        # Check different
        self.check_btn = QPushButton("Check")
        self.layout.addWidget(self.check_btn, 2, 2, 1, 1)
        self.check_btn.clicked.connect(self.check_diff)
        self.check_btn.setEnabled(False)

        # Load
        self.load_saliency_btn = QPushButton("Load Saliency")
        self.layout.addWidget(self.load_saliency_btn, 2, 3, 1, 1)
        self.load_saliency_btn.clicked.connect(self.load_check_instance)
        self.load_saliency_btn.setEnabled(True)


    def reset_viewer(self, mode='all') -> None:

        self.viewer.items = []
        # self.viewer.addItem(self.grid)
        if mode == 'all':
            self.viewer2.items = []
            # self.viewer2.addItem(self.grid)

    def show_instance(self):
        self.reset_viewer()
        single_data_info = self.data_info[self.index]
        self.data_dict = common.extract_points_from_pkl(single_data_info)
        mesh = common.get_points_mesh(self.data_dict['points'], 15)
        self.file_name_label.setText(single_data_info['path'])
        self.viewer.addItem(mesh)
        #  dropped points
        mesh = common.get_points_mesh(self.data_dict['dropped_points'], 15)
        self.viewer2.addItem(mesh)

    def load_kitti_instance(self):
        self.data_info = common.load_kitti_pkl()
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.inference_classific_btn.setEnabled(True)
        self.show_instance()

    @property
    def __len__(self):
        if hasattr(self, 'data_info'):
            return len(self.data_info)

    def check_index_overflow(self) -> None:

        totle_lenth = len(self.data_info)

        if self.index == -1:
            self.index = totle_lenth - 1

        if self.index >= totle_lenth:
            self.index = 0

    def decrement_index(self) -> None:

        if self.index != -1:
            self.index -= 1
            self.check_index_overflow()

        self.show_instance()

    def increment_index(self) -> None:

        if self.index != -1:

            self.index += 1
            self.check_index_overflow()

        self.show_instance()

    def predict(self):
        self.saliency_class.model_data_prepare(self.data_dict)
        results, dropped_results = self.saliency_class.inference()
        text_item = gl.GLTextItem(pos=[0,0,1], 
                                  text=str(common.CLASS_NUM[results.detach().cpu()]), 
                                  color=(255, 255, 255, 255), 
                                  font=QFont('Helvetica', 12))
        self.viewer.addItem(text_item)

        text_item = gl.GLTextItem(pos=[0,0,1], 
                            text=str(common.CLASS_NUM[dropped_results.detach().cpu()]), 
                            color=(255, 255, 255, 255), 
                            font=QFont('Helvetica', 12))
        self.viewer2.addItem(text_item)
        self.show_saliency_btn.setEnabled(True)
        self.check_btn.setEnabled(True)


    def show_saliency(self):
        self.reset_viewer(mode='1')
        saliency_score = self.saliency_class.get_saliency_scores()
        mesh = common.get_custom_colors(self.data_dict['points'], feature=saliency_score, size=15)
        self.viewer.addItem(mesh)

    def check_diff(self):
        import pickle
        output_path = '/home/alan/Desktop/pointMLP-pytorch/Saliency_KITTI/data/kitti/check_saliency.pkl'
        check_list = []
        save_num = 100
        for idx in range(len(self.data_info)):
            single_data_info = self.data_info[idx]
            data_dict = common.extract_points_from_pkl(single_data_info)
            self.saliency_class.model_data_prepare(data_dict)
            results, dropped_results = self.saliency_class.inference()
            if results != dropped_results:
                check_list.append(single_data_info)
            if len(check_list) > save_num:
                break

        with open(output_path, 'wb') as f:
            pickle.dump(check_list, f)
        print('Over')
        # self.load_saliency_btn.setEnabled(True)
        
    def load_check_instance(self):
        try:
            self.data_info = common.load_check_saliency_pkl()
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.inference_classific_btn.setEnabled(True)
            self.show_instance()
        except:
            raise NotImplementedError('No such file')
