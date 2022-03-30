#
# boostcamp AI Tech
# Trash Object Detection Competition
#


from mmcv import Config


if __name__ == '__main__':
    cfg = Config.fromfile('/opt/ml/detection/swin/configs/modified_swin_base.py')
    print(cfg.pretty_text)
