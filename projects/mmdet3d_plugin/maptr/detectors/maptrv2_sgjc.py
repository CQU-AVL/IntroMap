# projects/mmdet3d_plugin/maptr/detectors/maptrv2_sgjc.py
from mmdet.models import DETECTORS
from .maptrv2 import MapTRv2


@DETECTORS.register_module()
class MapTRv2SGJC(MapTRv2):
    def forward_pts_train(self, pts_feats, lidar_feat, gt_bboxes_3d, gt_labels_3d,
                          img_metas, gt_bboxes_ignore=None, prev_bev=None,
                          gt_depth=None, gt_seg_mask=None, gt_pv_seg_mask=None):

        # 1. 先跑 head，得到包含 ceh_preds 的 outs
        outs = self.pts_bbox_head(pts_feats, lidar_feat, img_metas, prev_bev)

        # 2. 原始 MapTRv2Head.loss 只接受这 4 个参数
        losses_pts = self.pts_bbox_head.loss(
            gt_bboxes_list=gt_bboxes_3d,      # 参数1
            gt_labels_list=gt_labels_3d,      # 参数2
            preds_dicts=outs,                 # 参数3  ← 关键！原始就是叫 preds_dicts
            gt_bboxes_ignore=gt_bboxes_ignore,
            # 注意：这里不要传 gt_seg_mask、gt_pv_seg_mask、img_metas（原始头根本不认识）
        )

        # 3. 原始 loss 返回的 dict 里只有 MapTR 的 loss
        #    现在我们把 outs 再传一次，让我们自己的 MapTRHeadWithSGJC 去计算 SGJC/GSCR loss
        if hasattr(self.pts_bbox_head, 'loss'):
            # 再次调用我们自己的 loss（带 extra_outputs）
            sgjc_losses = self.pts_bbox_head.loss(
                gt_bboxes_list=gt_bboxes_3d,
                gt_labels_list=gt_labels_3d,
                preds_dicts=outs,
                gt_bboxes_ignore=gt_bboxes_ignore,
                img_metas=img_metas,          # 我们自己的 head 需要这个
                extra_outputs=outs            # 关键！触发 SGJC + GSCR 计算
            )
            # 把所有 SGJC 相关 loss 合并到最终 dict
            losses_pts.update(sgjc_losses)

        return losses_pts