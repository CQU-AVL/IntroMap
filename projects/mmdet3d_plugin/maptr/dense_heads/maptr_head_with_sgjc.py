# file: projects/mmdet3d_plugin/maptr/dense_heads/maptr_head_with_sgjc.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmcv.utils import get_logger
from .maptrv2_head import MapTRv2Head
from .sgjc_head import SGJCHead
from .gscr_head import GSCR
from ..losses.sgjc_loss import sgjc_loss


@HEADS.register_module()
class MapTRHeadWithSGJC(MapTRv2Head):
    def __init__(self,
                 num_vec_one2one=70,
                 num_vec_one2many=300,
                 num_classes=4,
                 num_pts_per_vec=20,
                 # 添加 loss_consist
                 sgjc_loss_weights=dict(loss_geo=1.0, loss_nll=1.0, loss_sem_reg=0.05, loss_sem_cls=1.0,loss_sem_var=1.0,
                                        loss_consist=0.01),
                 sgjc_cross_encourage_weight=0.05,
                 gscr_geo_reg_weight=0.1,
                 gscr_sem_reg_weight=0.1,
                 **kwargs):
        super().__init__(num_vec_one2one=num_vec_one2one, num_vec_one2many=num_vec_one2many,
                         num_classes=num_classes, num_pts_per_vec=num_pts_per_vec, **kwargs)
        self.num_vec = num_vec_one2one + num_vec_one2many
        self.num_pts_per_vec = num_pts_per_vec

        self.sgjc_loss_weights = sgjc_loss_weights
        self.sgjc_cross_encourage_weight = sgjc_cross_encourage_weight
        self.gscr_geo_reg_weight = gscr_geo_reg_weight
        self.gscr_sem_reg_weight = gscr_sem_reg_weight

        if hasattr(self.transformer.decoder, 'layers'):
            for layer in self.transformer.decoder.layers:
                try:
                    setattr(layer, 'num_vec', self.num_vec)
                    setattr(layer, 'num_pts_per_vec', self.num_pts_per_vec)
                except Exception:
                    pass

        self.sgjc_head = SGJCHead(in_channels=self.embed_dims, num_classes=num_classes)
        # geo_in_dim=3 because Sigma_geo has 3 unique elements (xx, yy, xy)
        self.gscr = GSCR(dim=self.embed_dims, geo_in_dim=3, sem_in_dim=1)

    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False):
        outs = super().forward(mlvl_feats, lidar_feat, img_metas, prev_bev, only_bev)

        query_embed = None
        possible_keys = ['hs', 'all_query_embeddings', 'queries', 'query_feats']
        for key in possible_keys:
            if key in outs and outs[key] is not None:
                query_embed = outs[key][-1]
                break
        if query_embed is None: return outs

        B = len(img_metas) if isinstance(img_metas, list) else 1
        # Dimension handling... (Same as before)
        if query_embed.dim() == 2:
            B, NP, D = 1, query_embed.shape[0], query_embed.shape[1]
            query_embed = query_embed.unsqueeze(0)
        elif query_embed.dim() == 3:
            if query_embed.shape[0] != B and query_embed.shape[1] == B:
                query_embed = query_embed.permute(1, 0, 2)
            B, NP, D = query_embed.shape
        else:
            return outs

        P = self.num_pts_per_vec
        N = NP // P
        if NP % P != 0:
            N = NP // P
            query_embed = query_embed[:, :N * P, :]
        query_flat = query_embed.reshape(B * N * P, D)

        # 1. Pass 1 (Raw)
        sgjc_out_raw = self.sgjc_head(query_flat)

        # 2. GSCR (Active Modulation)
        geo_uncert = sgjc_out_raw.get('Sigma_geo', None)
        sem_uncert = sgjc_out_raw.get('semantic_var', None)

        # No Detach!
        shift_geo_feat, shift_sem_feat = self.gscr(geo_uncert, sem_uncert)
        query_corrected_flat = query_flat + shift_geo_feat + shift_sem_feat

        # 3. Main Head Override
        if hasattr(self, 'cls_heads') and hasattr(self, 'bbox_heads'):
            L = len(self.cls_heads)
            final_cls_head = self.cls_heads[L - 1]
            final_bbox_head = self.bbox_heads[L - 1]
            final_cls_score_corrected_flat = final_cls_head(query_corrected_flat)
            final_bbox_pred_corrected_flat = final_bbox_head(query_corrected_flat)
            cls_pred_reshaped = final_cls_score_corrected_flat.view(B, N, P, -1)
            final_cls_score_corrected = cls_pred_reshaped.mean(dim=2)
            try:
                orig_bbox_shape = outs['all_bbox_preds'][-1].shape
                if len(orig_bbox_shape) == 3:
                    final_bbox_pred_corrected = final_bbox_pred_corrected_flat.view(B, N, -1)
                    outs['all_bbox_preds'][-1] = final_bbox_pred_corrected
                    outs['all_cls_scores'][-1] = final_cls_score_corrected
            except Exception:
                pass

        # 4. Pass 2 (Final)
        sgjc_out_final = self.sgjc_head(query_corrected_flat)

        def r(t):
            if isinstance(t, torch.Tensor):
                return t.view(B, N, P, *t.shape[1:])
            return t

        outs["ceh_preds"] = {
            "mu_joint": r(sgjc_out_final["mu_joint"]),  # 3D
            "mu_xy": r(sgjc_out_final["mu_xy"]),
            "logits": r(sgjc_out_final["logits"]),
            "Sigma_joint": r(sgjc_out_final["Sigma_joint"]),
            "Sigma_geo": r(sgjc_out_final["Sigma_geo"]),
            "semantic_var": r(sgjc_out_final["semantic_var"]),
            "p_sem": r(sgjc_out_final["p_sem"]),
            "shift_geo_feat": r(shift_geo_feat),
            "shift_sem_feat": r(shift_sem_feat),

            # Save Raw Logits for Consistency Loss
            "logits_raw": r(sgjc_out_raw["logits"]),
        }
        return outs

    def loss(self, gt_bboxes_3d, gt_labels_3d, *args, **kwargs):
        extra_outputs = kwargs.pop('extra_outputs', None)
        ceh_preds_kw = kwargs.pop('ceh_preds', None)
        loss_dict = super().loss(gt_bboxes_3d, gt_labels_3d, *args, **kwargs)

        ceh_preds = None
        if ceh_preds_kw is not None:
            ceh_preds = ceh_preds_kw
        elif isinstance(extra_outputs, dict) and 'ceh_preds' in extra_outputs:
            ceh_preds = extra_outputs['ceh_preds']
        if ceh_preds is None and len(args) > 0 and isinstance(args[0], dict): ceh_preds = args[0].get('ceh_preds')
        if ceh_preds is None: return loss_dict

        # SGJC Loss
        try:
            sgjc_loss_dict = sgjc_loss(
                ceh_preds=ceh_preds,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                img_metas=kwargs.get('img_metas', None),
                matched_indices=None,
                weights=self.sgjc_loss_weights,
                cross_encourage_weight=self.sgjc_cross_encourage_weight
            )
        except Exception as e:
            sgjc_loss_dict = {}

        # GSCR Reg
        shift_geo = ceh_preds.get('shift_geo_feat')
        if shift_geo is not None: loss_dict['loss_sgjc_geo_reg'] = shift_geo.norm(
            dim=-1).mean() * self.gscr_geo_reg_weight
        shift_sem = ceh_preds.get('shift_sem_feat')
        if shift_sem is not None: loss_dict['loss_sgjc_sem_reg'] = shift_sem.norm(
            dim=-1).mean() * self.gscr_sem_reg_weight

        # Consistency Loss (Idea 3)
        logits_raw = ceh_preds.get('logits_raw')
        logits_final = ceh_preds.get('logits')
        if logits_raw is not None and logits_final is not None:
            # KL(P_raw || P_final)
            # Use log_softmax for raw, softmax for final target
            log_p_raw = F.log_softmax(logits_raw, dim=-1)
            p_final = F.softmax(logits_final.detach(), dim=-1)  # Target is fixed (corrected is better)

            loss_consist = F.kl_div(log_p_raw, p_final, reduction='batchmean')
            loss_dict['loss_sgjc_consist'] = loss_consist * self.sgjc_loss_weights.get('loss_consist', 2.0)

        safe_device = ceh_preds['mu_xy'].device
        for k, v in sgjc_loss_dict.items():
            if not isinstance(v, torch.Tensor): v = torch.tensor(v, device=safe_device)
            if v.dim() > 0: v = v.mean()
            if 'nll' in k: v = torch.clamp(v, max=20.0)
            loss_dict[f'loss_sgjc_{k}'] = v

        return loss_dict