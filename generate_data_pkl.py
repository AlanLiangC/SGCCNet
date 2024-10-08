import os
import pickle
import random
import numpy as np

data_path = 'data/kitti'
class_num = ['Car', 'Pedestrian', 'Cyclist']


least_points_num_for_train_val= {
    "Car" : 5,
    "Pedestrian": 5,
    "Cyclist": 5
}


# PKL for training and val
# For points < 5

def filt_data(pkl_path, output_path):
    all_data = []
    f = open(pkl_path, 'rb')
    data = pickle.load(f)
    for key, value in data.items():
        if key in class_num:
            points_num_lower_limit = least_points_num_for_train_val[key]
            for sample in value:
                if sample['num_points_in_gt'] > points_num_lower_limit:
                    all_data.append(sample)

    random.shuffle(all_data)
    with open(output_path, 'wb') as f:
        pickle.dump(all_data, f)

# PKL for Droppping
        
def saving_new_pkl(pkl_path):
    all_data = {}
    f = open(pkl_path, 'rb')
    data = pickle.load(f)
    for key, value in data.items():
        if key in class_num:
            sub_data = []
            points_num_lower_limit = least_points_num_for_train_val[key]
            for sample in value:
                if sample['difficulty'] < 2:
                    if sample['num_points_in_gt'] > points_num_lower_limit:
                        sample['path'] = 'saliency_gt_database/' + sample['path'].split('/')[-1]
                        dropped_path = os.path.join(data_path, 'saliency_gt_database/' + sample['path'].split('/')[-1])
                        dropped_points = np.fromfile(dropped_path, dtype=np.float32).reshape(-1, 4)
                        sample['num_points_in_gt'] = dropped_points.shape[0]
                sub_data.append(sample)
            all_data.update({
                key: sub_data
            })
        else:
            all_data.update({
                key: value
            })

    with open(new_pkl_path, 'wb') as f:
        pickle.dump(all_data, f)
if __name__ == "__main__":

    ori_pkl_path_train = 'data/kitti/kitti_dbinfos_train.pkl'
    filt_pkl_path_train = './data/kitti/kitti_filt_dbinfos_train.pkl'

    ori_pkl_path_val = 'data/kitti/kitti_dbinfos_val.pkl'
    filt_pkl_path_val = './data/kitti/kitti_filt_dbinfos_val.pkl'


    new_pkl_path = 'data/kitti/new_kitti_dbinfos_train.pkl'

    filt_data(ori_pkl_path_train, filt_pkl_path_train)
    filt_data(ori_pkl_path_val, filt_pkl_path_val)

    # saving_new_pkl()
