# Modified from
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/_base_/default_runtime.py

checkpoint_config = dict(max_keep_ckpts=100, interval=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            interval=100,
            init_kwargs=dict(project='trash_detection_nestiank',
                entity='bucket_interior'),
            log_artifact=False
        )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
