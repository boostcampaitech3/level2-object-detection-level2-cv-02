from mmcv import Config

cfg = Config.fromfile('/opt/ml/detection/swin/configs/modified_swin_base.py')

print(cfg.pretty_text)
