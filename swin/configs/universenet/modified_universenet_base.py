# Modified from
# https://github.com/shinya7y/UniverseNet/blob/master/configs/universenet/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco.py

pretrained = 'https://github.com/shinya7y/UniverseNet/releases/download/20.07/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco_20200716_epoch_24-1b9a1241.pth'  # noqa

_base_ = [
    '/opt/ml/detection/swin/configs/universenet/universenet101_gfl.py',
    '/opt/ml/detection/swin/configs/universenet/dataset_mstrain_480_960.py',
    '/opt/ml/detection/swin/configs/default_runtime.py',
    '/opt/ml/detection/swin/configs/universenet/schedule_1x.py'
]

model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=pretrained),
)

optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)

seed = 2022
gpu_ids = [0]

fp16 = dict(loss_scale=512.)
