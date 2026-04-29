# file: projects/mmdet3d_plugin/maptr/dense_heads/gscr_head.py
import torch
import torch.nn as nn


class GSCR(nn.Module):
    def __init__(self, dim=256, geo_in_dim=3, sem_in_dim=1, hidden=64):
        super().__init__()
        # geo_in_dim=3: [sigma_xx, sigma_yy, sigma_xy]
        # sem_in_dim=1: [sigma_sem]

        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim)
        )
        self.sem_mlp = nn.Sequential(
            nn.Linear(sem_in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim)
        )

        # Zero init for stability
        nn.init.constant_(self.geo_mlp[-1].weight, 0)
        nn.init.constant_(self.geo_mlp[-1].bias, 0)
        nn.init.constant_(self.sem_mlp[-1].weight, 0)
        nn.init.constant_(self.sem_mlp[-1].bias, 0)

    def forward(self, Sigma_geo, sem_var):
        """
        Args:
            Sigma_geo: [M, 2, 2] 几何协方差
            sem_var: [M] or [M, 1] 语义方差
        """
        # 1. 处理几何特征 [M, 3]
        geo_feat = self.extract_geo_features(Sigma_geo)

        # 2. 处理语义特征 [M] -> [M, 1]
        # 【核心修复】：确保最后维度是 1，以匹配 Linear(1, hidden)
        if sem_var.dim() == 1:
            sem_var = sem_var.unsqueeze(-1)  # [7400] -> [7400, 1]
        elif sem_var.dim() == 2 and sem_var.shape[0] == 1 and sem_var.shape[1] > 1:
            # 防止某些情况下输入变成了 [1, 7400]
            sem_var = sem_var.transpose(0, 1)

        shift_geo = self.geo_mlp(geo_feat)
        shift_sem = self.sem_mlp(sem_var)

        return shift_geo, shift_sem

    def extract_geo_features(self, Sigma_geo):
        # Flatten to unique elements: xx, yy, xy
        # Sigma is symmetric
        sigma_xx = Sigma_geo[..., 0, 0].unsqueeze(-1)
        sigma_yy = Sigma_geo[..., 1, 1].unsqueeze(-1)
        sigma_xy = Sigma_geo[..., 0, 1].unsqueeze(-1)
        # Concatenate: [..., 3]
        return torch.cat([sigma_xx, sigma_yy, sigma_xy], dim=-1)