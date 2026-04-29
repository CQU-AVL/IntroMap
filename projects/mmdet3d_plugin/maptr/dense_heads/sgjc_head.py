# file: projects/mmdet3d_plugin/maptr/dense_heads/sgjc_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SGJCHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=4, cholesky_init_log=-0.3, min_diag=1e-6, cross_init_scale=0.0):
        super().__init__()

        # 【核心修改 1】: 输出维度改为 3 (x, y, semantic_error_proxy)
        self.mlp_mu = nn.Linear(in_channels, 3)

        self.mlp_logits = nn.Linear(in_channels, num_classes)
        self.mlp_sem_var = nn.Linear(in_channels, 1)

        self.norm_feat = nn.LayerNorm(in_channels)
        self.mlp_cholesky = nn.Linear(in_channels, 6)  # 维持 3x3 矩阵 (6个参数)

        # Init
        try:
            self.mlp_cholesky.bias.data.fill_(cholesky_init_log)
        except Exception:
            with torch.no_grad():
                self.mlp_cholesky.bias.copy_(torch.full_like(self.mlp_cholesky.bias, cholesky_init_log))

        self.min_diag = float(min_diag)
        self.log_temp_raw = nn.Parameter(torch.tensor(0.0))
        self.temp_min = 0.1
        self.temp_max = 10.0
        self.num_classes = int(num_classes)

    def forward(self, x):
        M = x.shape[0]
        device, dtype = x.device, x.dtype

        # 1. 预测 3D 均值
        mu_raw = self.mlp_mu(x)  # [M, 3]

        # x, y 做 Sigmoid 归一化
        mu_xy = mu_raw[:, :2].sigmoid()

        # z (语义误差预测) 应该是正数 (类似于 Loss)，用 Softplus 激活
        # 这代表模型预测的 "Expected Cross Entropy Loss"
        mu_sem_err = F.softplus(mu_raw[:, 2:3])

        # 拼接回 [M, 3]
        mu_joint = torch.cat([mu_xy, mu_sem_err], dim=-1)

        # 2. Logits & Uncertainty
        logits_raw = self.mlp_logits(x)
        sem_var_raw = self.mlp_sem_var(x).squeeze(-1)
        semantic_uncertainty_var = F.softplus(sem_var_raw) + 1e-6

        temp = self.temp_min + (self.temp_max - self.temp_min) * torch.sigmoid(self.log_temp_raw)
        p_sem = F.softmax(logits_raw / temp, dim=-1)

        # 3. 协方差矩阵 (3x3)
        x_norm = self.norm_feat(x)
        chol_params = self.mlp_cholesky(x_norm)
        l00, l10, l11, l20, l21, l22 = torch.chunk(chol_params, 6, dim=-1)

        l00 = F.softplus(l00).squeeze(-1) + self.min_diag
        l11 = F.softplus(l11).squeeze(-1) + self.min_diag
        l22 = F.softplus(l22).squeeze(-1) + self.min_diag
        l10 = l10.squeeze(-1)
        l20 = l20.squeeze(-1)
        l21 = l21.squeeze(-1)

        L = torch.zeros(M, 3, 3, device=device, dtype=dtype)
        L[:, 0, 0] = l00
        L[:, 1, 0] = l10
        L[:, 1, 1] = l11
        L[:, 2, 0] = l20
        L[:, 2, 1] = l21
        L[:, 2, 2] = l22

        Sigma_joint = L @ L.transpose(-1, -2)  # [M, 3, 3]
        Sigma_geo = Sigma_joint[:, :2, :2]  # [M, 2, 2]

        return {
            'mu_joint': mu_joint,  # [M, 3] (x, y, sem_err)
            'mu_xy': mu_xy,  # [M, 2] (兼容旧代码)
            'Sigma_geo': Sigma_geo,
            'Sigma_joint': Sigma_joint,
            'logits': logits_raw,
            'p_sem': p_sem,
            'semantic_var': semantic_uncertainty_var,
            'temp': temp
        }