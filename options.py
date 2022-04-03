#
# boostcamp AI Tech
# Trash Object Detection Competition
#


from mmcv import Config

from pycocotools.coco import COCO
import pandas as pd

RUN_NAME = 'Swin_HeavyAugs2_Epochs42_60'
CONFIG_PATH = '/opt/ml/detection/swin/configs/heavy_augs/modified_swin_base_heavy_augs.py'
CONFIG_PATH_LOW_THR = '/opt/ml/detection/swin/configs/thr_down/modified_swin_base_heavy_augs_thr_down.py'


def get_cfg(loc: str, run: str, epochs: int):
    cfg = Config.fromfile(loc)
    cfg.log_config.hooks[1].init_kwargs.name = run
    cfg.runner.max_epochs = epochs
    return cfg


def make_predictions(output, cfg, loc: str):
    prediction_strings = []
    file_names = []

    coco = COCO(cfg.data.test.ann_file)

    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(10):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(o[2]) + ' ' + str(o[3]) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(loc, index=None)
