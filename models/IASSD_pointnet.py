import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from typing import List

from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_modules import _PointnetSAModuleBase
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_utils

class PointnetSAModuleMSG_WithSampling(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with specific downsampling and multiscale grouping """

    def __init__(self, *,
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],                 
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 aggregation_mlp: List[int],
                 confidence_mlp: List[int],
                 num_class):
        
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        norm_func = partial(nn.BatchNorm2d, eps=1e-5, momentum= 0.1)


        out_channels = 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]

            self.groupers.append(
                pointnet2_utils.QueryAndGroup(
                    radius, nsample, use_xyz=use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1],
                              kernel_size=1, bias=False),
                    norm_func(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method

        if (aggregation_mlp is not None) and (len(aggregation_mlp) != 0) and (len(self.mlps) > 0):
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_layer = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_layer = None

        if (confidence_mlp is not None) and (len(confidence_mlp) != 0):
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              confidence_mlp[k], kernel_size=1, bias=False),
                    # nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, num_class, kernel_size=1, bias=True),
            )
            self.confidence_layers = nn.Sequential(*shared_mlp)
        else:
            self.confidence_layers = None

        shared_mlp = []
        classific_mlp = [32,16]
        for k in range(len(classific_mlp)):
            shared_mlp.extend([
                nn.Linear(out_channels,
                            classific_mlp[k], bias=False),
                # nn.BatchNorm1d(classific_mlp[k]),
                nn.ReLU()
            ])
            out_channels = classific_mlp[k]
        shared_mlp.append(
            nn.Linear(out_channels, num_class, bias=True),
        )
        self.classific_layers = nn.Sequential(*shared_mlp)


    def forward(self, xyz: torch.Tensor, 
                features: torch.Tensor = None):
        
        new_features_list = []
        sampled_idx_list = []

        new_xyz = xyz
        if len(self.groupers) > 0:
            for i in range(len(self.groupers)):
                new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                if self.pool_method == 'max_pool':
                    new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                elif self.pool_method == 'avg_pool':
                    new_features = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                else:
                    raise NotImplementedError

                new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                new_features_list.append(new_features)

            new_features = torch.cat(new_features_list, dim=1)

            if self.aggregation_layer is not None:
                new_features = self.aggregation_layer(new_features)
        else:
            new_features = pointnet2_utils.gather_operation(features, sampled_idx_list).contiguous()

        new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(2)]
                    )
        result = self.classific_layers(new_features.squeeze(dim=-1))
        return result