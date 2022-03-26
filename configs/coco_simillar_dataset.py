# Modified from
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/_base_/datasets/coco_detection.py

dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/' # Modified
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True), # Modified
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32), # Modified
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512), # Modified
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32), # Modified
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", # Modified
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing") # Modified
data = dict(
    samples_per_gpu=8, # Modified
    workers_per_gpu=2, # Modified
    train=dict(
        type=dataset_type,
        classes=classes, # Modified
        ann_file=data_root + 'train.json', # Modified
        img_prefix=data_root, # Modified
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes, # Modified
        ann_file=data_root + 'val.json', # Modified
        img_prefix=data_root, # Modified
        pipeline=test_pipeline, # Modified
        test_mode=True), # Modified
    test=dict(
        type=dataset_type,
        classes=classes, # Modified
        ann_file=data_root + 'test.json', # Modified
        img_prefix=data_root, # Modified
        pipeline=test_pipeline, # Modified
        test_mode=True)) # Modified
# evaluation = dict(metric='bbox', classwise=True) # Modified
