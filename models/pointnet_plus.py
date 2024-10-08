import torch.nn as nn
import torch.nn.functional as F
from .IASSD_pointnet import PointnetSAModuleMSG_WithSampling

class PointNetPlus(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        sa_config = config.SA_CONFIG
        channel_in = 1
        self.aggregation_mlps = sa_config.get('AGGREGATION_MLPS', None)
        self.confidence_mlps = sa_config.get('CONFIDENCE_MLPS', None)
        self.num_class = 3

        mlps = sa_config.MLPS[0].copy()
        channel_out = 0
        for idx in range(mlps.__len__()):
            mlps[idx] = [channel_in] + mlps[idx]
            channel_out += mlps[idx][-1]

        if self.aggregation_mlps and self.aggregation_mlps[0]:
            aggregation_mlp = self.aggregation_mlps[0].copy()
            if aggregation_mlp.__len__() == 0:
                aggregation_mlp = None
            else:
                channel_out = aggregation_mlp[-1]
        else:
            aggregation_mlp = None

        if self.confidence_mlps and self.confidence_mlps[0]:
            confidence_mlp = self.confidence_mlps[0].copy()
            if confidence_mlp.__len__() == 0:
                confidence_mlp = None
        else:
            confidence_mlp = None

        self.backbone = PointnetSAModuleMSG_WithSampling(
            radii=sa_config.RADIUS_LIST[0],
            nsamples=sa_config.NSAMPLE_LIST[0],
            mlps=mlps,
            use_xyz=True,                                                
            aggregation_mlp=aggregation_mlp,
            confidence_mlp=confidence_mlp,
            num_class = self.num_class
        )

    def break_up_pc(self, pc):
        pc = F.pad(pc, (1,0), 'constant', 0)
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features


    def forward(self, data_dict):

        batch_size = 1
        points = data_dict['points']

        batch_idx, xyz, features = self.break_up_pc(points.squeeze())

        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None ###
        results = self.backbone(xyz, features)

        return results



