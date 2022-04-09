# Modified from
# https://github.com/shinya7y/UniverseNet/blob/master/configs/_base_/schedules/schedule_1x.py

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 15])
runner = dict(type='EpochBasedRunner', max_epochs=36)
