# Modified from
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/_base_/schedules/schedule_1x.py

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 15])
runner = dict(type='EpochBasedRunner', max_epochs=60)
