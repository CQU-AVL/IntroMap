# configs/maptrv2_sgjc_nuscenes_mini.py
base = [
'../datasets/custom_nus-3d.py',
'../base/default_runtime.py'
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
voxel_size = [0.15, 0.15, 20.0]
dbound = [1.0, 35.0, 0.5]
grid_config = {
'x': [-30.0, -30.0, 0.15],
'y': [-15.0, -15.0, 0.15],
'z': [-10, 10, 20],
'depth': [1.0, 35.0, 0.5],
}
img_norm_cfg = dict(
mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = [
'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
map_classes = ['divider', 'ped_crossing', 'boundary', 'centerline']
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag = True
num_map_classes = len(map_classes)
input_modality = dict(
use_lidar=False,
use_camera=True,
use_radar=False,
use_map=False,
use_external=True)
dim = 256
pos_dim = dim // 2
ffn_dim = dim * 2
num_levels = 1
bev_h_ = 200
bev_w_ = 100
queue_length = 1
aux_seg_cfg = dict(
use_aux_seg=True,
bev_seg=True,
pv_seg=True,
seg_classes=1,
feat_down_sample=32,
pv_thickness=1,
)
model = dict(
type='MapTRv2',
use_grid_mask=True,
video_test_mode=False,
pretrained=dict(img='/mnt/f/CQU/MapTR-maptrv2/ckpts/resnet50-19c8e357.pth'),
img_backbone=dict(
type='ResNet',
depth=50,
num_stages=4,
out_indices=(3,),
frozen_stages=1,
norm_cfg=dict(type='BN', requires_grad=False),
norm_eval=True,
style='pytorch'),
img_neck=dict(
type='FPN',
in_channels=[2048],
out_channels=dim,
start_level=0,
add_extra_convs='on_output',
num_outs=num_levels,
relu_before_extra_convs=True),
pts_bbox_head=dict(
type='MapTRHeadWithSGJC',
bev_h=bev_h_,
bev_w=bev_w_,
num_query=300,
num_vec_one2one=20,           # mini 专用
num_vec_one2many=40,          # mini 专用
k_one2many=6,
num_pts_per_vec=fixed_ptsnum_per_pred_line,
num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
dir_interval=1,
query_embed_type='instance_pts',
transform_method='minmax',
gt_shift_pts_pattern='v2',
num_classes=num_map_classes,
in_channels=dim,
custom_loss=dict(type='sgjc_loss',loss_weight=1.0),
sync_cls_avg_factor=True,
with_box_refine=True,
as_two_stage=False,
code_size=2,
code_weights=[1.0, 1.0, 1.0, 1.0],
aux_seg=aux_seg_cfg,
transformer=dict(
type='MapTRPerceptionTransformer',
rotate_prev_bev=True,
use_shift=True,
use_can_bus=True,
embed_dims=dim,
encoder=dict(
type='LSSTransform',
in_channels=dim,
out_channels=dim,
feat_down_sample=32,
pc_range=point_cloud_range,
voxel_size=voxel_size,
dbound=dbound,
downsample=2,
loss_depth_weight=3.0,
depthnet_cfg=dict(use_dcn=False, with_cp=False, aspp_mid_channels=96),
grid_config=grid_config,
),
decoder=dict(
type='MapTRDecoder',
num_layers=6,
return_intermediate=True,
transformerlayers=dict(
type='DecoupledDetrTransformerDecoderLayer',
num_vec=None,
num_pts_per_vec=fixed_ptsnum_per_pred_line,
attn_cfgs=[
dict(type='MultiheadAttention', embed_dims=dim, num_heads=8, dropout=0.1),
dict(type='MultiheadAttention', embed_dims=dim, num_heads=8, dropout=0.1),
dict(type='CustomMSDeformableAttention', embed_dims=dim, num_levels=1),
],
feedforward_channels=ffn_dim,
ffn_dropout=0.1,
operation_order=('self_attn', 'norm', 'self_attn', 'norm', 'cross_attn', 'norm',
'ffn', 'norm')
)
)
),
bbox_coder=dict(
type='MapTRNMSFreeCoder',
post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
pc_range=point_cloud_range,
max_num=50,
voxel_size=voxel_size,
num_classes=num_map_classes
),
positional_encoding=dict(
type='LearnedPositionalEncoding',
num_feats=pos_dim,
row_num_embed=bev_h_,
col_num_embed=bev_w_,
),
loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
loss_bbox=dict(type='L1Loss', loss_weight=0.0),
loss_iou=dict(type='GIoULoss', loss_weight=0.0),
loss_pts=dict(type='PtsL1Loss', loss_weight=5.0),  # 必须是 L1
loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
loss_seg=dict(type='SimpleLoss', pos_weight=4.0, loss_weight=1.0),
loss_pv_seg=dict(type='SimpleLoss', pos_weight=1.0, loss_weight=2.0),
),
train_cfg=dict(pts=dict(
grid_size=[512, 512, 1],
voxel_size=voxel_size,
point_cloud_range=point_cloud_range,
out_size_factor=4,
assigner=dict(
type='MapTRAssigner',
cls_cost=dict(type='FocalLossCost', weight=2.0),
reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
pts_cost=dict(type='OrderedPtsL1Cost', weight=5),
pc_range=point_cloud_range
)
))
)
dataset_type = 'CustomNuScenesOfflineLocalMapDataset'
data_root = '/mnt/f/CQU/MapTR-maptrv2/data/nuscenes'
file_client_args = dict(backend='disk')
train_pipeline = [
dict(type='LoadMultiViewImageFromFiles', to_float32=True),
dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
dict(type='PhotoMetricDistortionMultiViewImage'),
dict(type='NormalizeMultiviewImage', **img_norm_cfg),
dict(type='LoadPointsFromFile',
coord_type='LIDAR',
load_dim=5,
use_dim=5,
file_client_args=file_client_args),
dict(type='CustomPointToMultiViewDepth', downsample=1, grid_config=grid_config),
dict(type='PadMultiViewImageDepth', size_divisor=32),
dict(type='DefaultFormatBundle3D',
with_gt=False, with_label=False, class_names=map_classes),
dict(type='CustomCollect3D', keys=['img', 'gt_depth'])  # 关键：gt_depth 包含 gt_pts
]
test_pipeline = [
dict(type='LoadMultiViewImageFromFiles', to_float32=True),
dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
dict(type='NormalizeMultiviewImage', **img_norm_cfg),
dict(
type='MultiScaleFlipAug3D',
img_scale=(1600, 900),
pts_scale_ratio=1,
flip=False,
transforms=[
dict(type='PadMultiViewImage', size_divisor=32),
dict(type='DefaultFormatBundle3D', with_gt=False, with_label=False, class_names=map_classes),
dict(type='CustomCollect3D', keys=['img'])
]
)
]
data = dict(
    samples_per_gpu=1,      # 移到这里
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_map_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        aux_seg=aux_seg_cfg,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        queue_length=queue_length,
        box_type_3d='LiDAR'
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_map_infos_temporal_val.pkl',
        map_ann_file=data_root + '/nuscenes_map_anns_val.json',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        classes=class_names,
        modality=input_modality
        # 删除 samples_per_gpu=1
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_map_infos_temporal_val.pkl',
        map_ann_file=data_root + '/nuscenes_map_anns_val.json',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        classes=class_names,
        modality=input_modality
        # 删除 samples_per_gpu
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)
optimizer = dict(
type='AdamW',
lr=1e-4,  # mini 专用
paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
weight_decay=0.01
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
policy='CosineAnnealing',
warmup='linear',
warmup_iters=500,
warmup_ratio=1.0 / 3,
min_lr_ratio=1e-3
)
total_epochs = 12  # mini 收敛快
evaluation = dict(interval=2, pipeline=test_pipeline, metric='chamfer',
save_best='NuscMap_chamfer/mAP', rule='greater')
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
interval=10,  # mini 建议频繁打印
hooks=[
dict(type='TextLoggerHook'),
dict(type='TensorboardLoggerHook')
]
)
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(max_keep_ckpts=1, interval=2)
find_unused_parameters = True

log_level = 'INFO'
workflow = [('train', 1), ('val', 1)]
resume_from = None
load_from = None