import models as models
import argparse

import os
from utils import cfg, cfg_from_yaml_file, load_data_to_gpu
from helper import cal_loss

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as torch_data

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, default='20240319101035-2459', help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='PointNetPlus', help='model name [default: pointnet_cls]')

    args = parser.parse_args()

    cfg_from_yaml_file('SK.yaml', cfg)

    return args, cfg

class DemoDataset(torch_data.Dataset):
    def __init__(self, data_dict):
        self.data_list = [data_dict]

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        return data_dict

class Salency_Class():
    def __init__(self):
        super().__init__()
        self.criterion = cal_loss

    def model_data_prepare(self, data_dict):

        # Dataset
        self.demo_dataloader = DemoDataset(data_dict)
        # self.demo_dataloader = torch_data.DataLoader(demo_dataset, num_workers=1,
        #                     batch_size=1, shuffle=False, drop_last=False)
        # Model

        device = 'cuda'
        args, cfg = parse_args()
        message = "-"+args.msg
        args.checkpoint = 'checkpoints/' + args.model + message

        print('==> Building model..')
        self.model = models.__dict__[args.model](cfg.MODEL)
        self.model = self.model.to(device)
        checkpoint_path = os.path.join(args.checkpoint, 'best_checkpoint.pth')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['net'])

    def inference(self):
        self.model.train()
        # with torch.no_grad():
        for batch_idx, batch_dict in enumerate(self.demo_dataloader):
            load_data_to_gpu(batch_dict)
            label = batch_dict['label']
            points = batch_dict['points']
            points.requires_grad = True
            batch_dict['points'] = points

            logits = self.model(batch_dict)
            self.loss = self.criterion(logits, label.long())
            self.gradients = torch.autograd.grad(self.loss, points, retain_graph=True)[0]
            self.data_dict = batch_dict

            temp_batch_dict = dict(
                points = batch_dict['dropped_points']
            )
            dropped_logits = self.model(temp_batch_dict)

        return logits.argmax(), dropped_logits.argmax()

    def get_saliency_scores(self):
        points = self.data_dict['points']
        # ALL
        r = torch.pow(torch.sum(torch.pow(points, 2),dim = 1), 0.5)
        delta_r = torch.sum(self.gradients*points,dim = 1)
        s = - delta_r * r
        return s
    
