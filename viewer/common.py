import pyqtgraph.opengl as gl
from PyQt5.QtCore import *

from pathlib import Path
import numpy as np
import pickle
import matplotlib.cm as cm
import matplotlib as mpl

import torch

ROOT_DIR = (Path(__file__).resolve().parent / '../data').resolve()

CLASS_NUM = ['Car', 'Pedestrian', 'Cyclist']

LABEL_MAPPING = {
    value : i for i, value in enumerate(CLASS_NUM)
}

class AL_viewer(gl.GLViewWidget):
    
    def __init__(self, parent=None, devicePixelRatio=None, rotationMethod='euler'):
        super().__init__(parent, devicePixelRatio, rotationMethod)

        self.noRepeatKeys = [Qt.Key.Key_W, Qt.Key.Key_S, Qt.Key.Key_A, Qt.Key.Key_D, Qt.Key.Key_Q, Qt.Key.Key_E,
            Qt.Key.Key_Right, Qt.Key.Key_Left, Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown]
        
        self.speed = 1

    def reset(self):
        super().reset()
        self.setBackgroundColor('k')
        
    def evalKeyState(self):
        vel_speed = 10 * self.speed 
        if len(self.keysPressed) > 0:
            for key in self.keysPressed:
                if key == Qt.Key.Key_Right:
                    self.orbit(azim=-self.speed, elev=0)
                elif key == Qt.Key.Key_Left:
                    self.orbit(azim=self.speed, elev=0)
                elif key == Qt.Key.Key_Up:
                    self.orbit(azim=0, elev=-self.speed)
                elif key == Qt.Key.Key_Down:
                    self.orbit(azim=0, elev=self.speed)
                elif key == Qt.Key.Key_A:
                    self.pan(vel_speed * self.speed, 0, 0, 'view-upright')
                elif key == Qt.Key.Key_D:
                    self.pan(-vel_speed, 0, 0, 'view-upright')
                elif key == Qt.Key.Key_W:
                    self.pan(0, vel_speed, 0, 'view-upright')
                elif key == Qt.Key.Key_S:
                    self.pan(0, -vel_speed, 0, 'view-upright')
                elif key == Qt.Key.Key_Q:
                    self.pan(0, 0, vel_speed, 'view-upright')
                elif key == Qt.Key.Key_E:
                    self.pan(0, 0, -vel_speed, 'view-upright')
                elif key == Qt.Key.Key_PageUp:
                    pass
                elif key == Qt.Key.Key_PageDown:
                    pass
                self.keyTimer.start(16)
        else:
            self.keyTimer.stop()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m, centroid

def load_kitti_pkl():
    pkl_path = 'data/kitti/kitti_saliency_dbinfos_train.pkl'
    f = open(pkl_path, 'rb')
    data_info = pickle.load(f)
    print('The size of data is %d' % (len(data_info)))
    return data_info

def load_check_saliency_pkl():
    pkl_path = 'data/kitti/check_saliency.pkl'
    f = open(pkl_path, 'rb')
    data_info = pickle.load(f)
    print('The size of data is %d' % (len(data_info)))
    return data_info

def get_lidar(path):
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)

def extract_points_from_pkl(insrtance_dict):
    lidar_path = 'kitti/' + (insrtance_dict['path'])
    lidar_path = ROOT_DIR.joinpath(Path(lidar_path))
    points = get_lidar(lidar_path)
    points[:, 0:3], m, centroid = pc_normalize(points[:, 0:3])
    # dropped points
    dropped_lidar_path = 'kitti/' + (insrtance_dict['drop_path'])
    dropped_lidar_path = ROOT_DIR.joinpath(Path(dropped_lidar_path))
    dropped_points = get_lidar(dropped_lidar_path)
    dropped_points[:,:3] = (dropped_points[:,:3] - centroid) / m

    data_dict = dict(
        points = points,
        dropped_points = dropped_points,
        label = np.array(LABEL_MAPPING[insrtance_dict['name']]).astype(np.int64), 
        sample_path = lidar_path
    )

    return data_dict

def get_points_mesh(points, size, colors = None):

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if colors == None:
        colors = np.ones([points.shape[0], 4])
        colors[:,3] = 1
    else:
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()

    mesh = gl.GLScatterPlotItem(pos=np.asarray(points[:, 0:3]), size=size, color=colors)

    return mesh


def normalize_feature(feature):
    feature_min = feature.min()
    feature_max = feature.max()

    feature = (feature - feature_min) / (feature_max - feature_min + 1e-5)

    return feature

def tensor2ndarray(value) -> np.ndarray:

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return value

def grading_feature(feature, num_grade = 10):
    grads = feature.new_zeros(feature.shape)
    L = feature.shape[0]
    split_num = L // num_grade
    for i in range(num_grade):
        index = torch.topk(feature, L - i*split_num, -1)[1]
        grads[index] = (i+1) / num_grade
    return tensor2ndarray(grads)

def get_custom_colors(xyz, feature, size=10):

    feature = grading_feature(feature)

    if isinstance(xyz, torch.Tensor):
        xyz = tensor2ndarray(xyz)
    if isinstance(feature, torch.Tensor):
        feature = tensor2ndarray(feature)

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.jet 
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(feature)
    colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
    colors[:, 3] = 0.5

    mesh = gl.GLScatterPlotItem(pos=np.asarray(xyz[:, 0:3]), size=size, color=colors)
    return mesh

