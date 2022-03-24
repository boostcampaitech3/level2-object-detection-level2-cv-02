# Modified from
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/_base_/default_runtime.py

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', # Modified
            interval=100, # Modified
            init_kwargs=dict(project='trash_detection_nestiank', # Modified
                entity='bucket_interior'), # Modified
            log_artifact=False # Modified
        ) # Modified
    ]) # Modified
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
