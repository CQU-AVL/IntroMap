# projects/mmdet3d_plugin/maptr/dense_heads/maptr_head_with_sgjc.py

from .sgjc_head import SGJCHead
from ..losses.sgjc_loss import sgjc_loss
from .maptrv2_head import MapTRv2Head
from mmdet.models import HEADS
import torch
import torch.nn as nn
from torch.nn import init as nn_init

@HEADS.register_module()
class MapTRHeadWithSGJC(MapTRv2Head):
    def __init__(self, **kwargs):
        num_vec_one2one = kwargs.get('num_vec_one2one', 70)
        num_vec_one2many = kwargs.get('num_vec_one2many', 300)
        total_vec = num_vec_one2one + num_vec_one2many
        kwargs['num_vec'] = total_vec
        kwargs['num_vec_one2one'] = num_vec_one2one
        kwargs['num_vec_one2many'] = num_vec_one2many

        super().__init__(**kwargs)

        self.num_vec_one2one = num_vec_one2one
        self.num_vec_one2many = num_vec_one2many
        self.num_vec = total_vec
        self.num_pts_per_vec = 20

        if hasattr(self, 'transformer') and hasattr(self.transformer, 'decoder'):
            dec = self.transformer.decoder
            for attr in ['num_vec', 'num_vec_one2one', 'num_vec_one2many', 'num_pts_per_vec']:
                setattr(dec, attr, getattr(self, attr))

        self.sgjc_head = SGJCHead(
            in_channels=self.embed_dims,
            num_classes=kwargs.get('num_classes', 4)
        )
        self.query_proj = None

    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False):
        """Forward function - 完全适配 MapTRv2 训练 + 推理"""

        # 关键修复 1：确保 decoder 参数在 super().forward() 之前设置！
        if hasattr(self.transformer, 'decoder'):
            dec = self.transformer.decoder
            dec.num_vec = self.num_vec
            dec.num_pts_per_vec = self.num_pts_per_vec
            dec.num_vec_one2one = self.num_vec_one2one
            dec.num_vec_one2many = self.num_vec_one2many
            # 不要重复设置！
            # for attr in [...] setattr(dec, attr, ...)
            for layer in dec.layers:
                layer.num_vec = self.num_vec
                layer.num_pts_per_vec = self.num_pts_per_vec
                layer.self_attn_mask = None

        # 关键修复 2：推理时，prev_bev 为 None 时，强制设置 num_query
        if not self.training and prev_bev is None:
            # 强制让父类知道 num_query = num_vec
            prev_bev = torch.zeros(1, self.num_vec * self.num_pts_per_vec, 256, device=mlvl_feats[0].device)

        # 调用父类 forward
        outs_dict = super().forward(mlvl_feats, lidar_feat, img_metas, prev_bev, only_bev)

        # CEH 预测
        pts_preds = outs_dict['all_pts_preds'][-1]  # [bs, num_query, num_pts, 2]
        bs, num_query, num_pts, _ = pts_preds.shape
        query_flat = pts_preds.flatten(0, 2)  # [N, 2]

        if self.query_proj is None:
            self.query_proj = nn.Linear(2, self.embed_dims).to(query_flat.device)
            nn_init.xavier_uniform_(self.query_proj.weight)
            nn_init.constant_(self.query_proj.bias, 0)

        query_embed = self.query_proj(query_flat)
        sgjc_preds = self.sgjc_head(query_embed)

        # 训练 vs 推理 统一返回格式
        if self.training:
            outs_dict['ceh_preds'] = sgjc_preds
            return outs_dict
        else:
            new_prev_bev = outs_dict.get('prev_bev', prev_bev)
            bbox_pts_list = outs_dict['all_pts_preds']
            return new_prev_bev, bbox_pts_list  # 必须返回 list！

    def loss(self, *args, **kwargs):
        outs_dict = args[0]
        gt_bboxes_3d = args[1]
        gt_labels_3d = args[2]
        gt_pts_list = args[3]
        img_metas = args[4] if len(args) > 4 else kwargs.get('img_metas')

        loss_dict = super().loss(*args, **kwargs)

        if 'ceh_preds' in outs_dict:
            try:
                ceh_loss_dict = sgjc_loss(outs_dict['ceh_preds'], gt_pts_list, img_metas)
                for k, v in ceh_loss_dict.items():
                    loss_dict[f'ceh_{k}'] = v
            except Exception as e:
                print(f"[CEH Loss Warning] {e}")

        return loss_dict