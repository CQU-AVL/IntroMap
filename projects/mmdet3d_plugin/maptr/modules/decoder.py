# file: projects/mmdet3d_plugin/maptr/modules/decoder.py

import torch
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      POSITIONAL_ENCODING,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer."""

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapTRDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


@TRANSFORMER_LAYER.register_module()
class DecoupledDetrTransformerDecoderLayer(BaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 num_vec=50,
                 num_pts_per_vec=20,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(DecoupledDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 8
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f'Use same attn_mask in all attentions in {self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn

        # 保存固定参数
        num_pts_per_vec = self.num_pts_per_vec

        # 【注意】不再使用 self.num_vec，所有地方都必须动态计算

        for layer in self.operation_order:
            if layer == 'self_attn':
                # ============================================================
                # 【核心修复区域】每次 Reshape 前都重新计算 actual_num_vec
                # ============================================================

                # 获取当前 query 的形状信息
                n_tokens, n_batch, n_dim = query.shape
                # 动态计算：总 Token 数 / 每个 Vector 的点数 = 实际 Vector 数
                actual_num_vec = n_tokens // num_pts_per_vec

                if attn_index == 0:
                    # --- Intra-instance attention (点之间 Attention) ---

                    # 1. 变换形状：[Num_Vec * P, B, D] -> [Num_Vec, P, B, D] -> [Num_Vec * P, B, D] (逻辑上的view)
                    # 此处根据 MapTR 逻辑，可能是要 flatten(1, 2) 把 P 和 B 混在一起做 attention

                    query = query.view(actual_num_vec, num_pts_per_vec, n_batch, n_dim).flatten(1, 2)
                    query_pos = query_pos.view(actual_num_vec, num_pts_per_vec, n_batch, n_dim).flatten(1, 2)

                    temp_key = temp_value = query

                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=getattr(self, 'self_attn_mask', None),
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)

                    # 2. 还原形状
                    # 这里的 flatten(0, 1) 是要把 [Num_Vec, P*B, D] 变回去？
                    # 不，attn 输出 shape 和 query 输入一致。
                    # 原代码这里可能是 query.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(0, 1)

                    # 我们使用 actual_num_vec
                    query = query.view(actual_num_vec, num_pts_per_vec, n_batch, n_dim).flatten(0, 1)
                    query_pos = query_pos.view(actual_num_vec, num_pts_per_vec, n_batch, n_dim).flatten(0, 1)

                    attn_index += 1
                    identity = query

                else:
                    # --- Inter-instance attention (Vector 之间 Attention) ---

                    # 1. 变换形状
                    # 原代码逻辑：query.view(num_vec, num_pts_per_vec, n_batch, n_dim).permute(1, 0, 2, 3).flatten(1, 2)

                    query = query.view(actual_num_vec, num_pts_per_vec, n_batch, n_dim).permute(1, 0, 2,
                                                                                                3).contiguous().flatten(
                        1, 2)
                    query_pos = query_pos.view(actual_num_vec, num_pts_per_vec, n_batch, n_dim).permute(1, 0, 2,
                                                                                                        3).contiguous().flatten(
                        1, 2)

                    temp_key = temp_value = query

                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=None,
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)

                    # 2. 还原形状 (这里就是你之前报错的 Line 219)
                    # 原代码：query.view(num_pts_per_vec, num_vec, n_batch, n_dim)...

                    # 必须使用 actual_num_vec !
                    query = query.view(num_pts_per_vec, actual_num_vec, n_batch, n_dim).permute(1, 0, 2,
                                                                                                3).contiguous().flatten(
                        0, 1)
                    query_pos = query_pos.view(num_pts_per_vec, actual_num_vec, n_batch, n_dim).permute(1, 0, 2,
                                                                                                        3).contiguous().flatten(
                        0, 1)

                    attn_index += 1
                    identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=None,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query