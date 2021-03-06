# Modified from
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/_base_/datasets/coco_detection.py

dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)
train_pipeline = [
    # RandomCrop, RandomFlip, PhotoMetricDistortion
    # Removed: CutOut
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(500, 500)),
    dict(type='RandomFlip', flip_ratio=[0.25, 0.25, 0.25], direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PhotoMetricDistortion'),
    # dict(type='CutOut', n_holes=1, cutout_ratio=[(0.1, 0.1), (0.15, 0.1), (0.1, 0.15), (0.15, 0.15)]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=True,
        transforms=[
            # RandomFlip
            # Removed: PhotoMetricDistortion
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=[0.25, 0.25, 0.25], direction=['horizontal', 'vertical', 'diagonal']),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='PhotoMetricDistortion'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True))

# evaluation = dict(metric='bbox', classwise=True)
