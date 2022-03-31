#
# boostcamp AI Tech
# Trash Object Detection Competition
#


from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import train_detector, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmcv.parallel import MMDataParallel
from pycocotools.coco import COCO

import os
import pandas as pd
import wandb


def get_cfg(loc: str, run: str, epochs: int):
    cfg = Config.fromfile(loc)
    cfg.checkpoint_config = dict(max_keep_ckpts=100, interval=2)
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.log_config.hooks[1].init_kwargs.name = run
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=epochs)
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


if __name__ == '__main__':
    # Experimental Hotfix
    os.environ["WANDB_START_METHOD"] = "thread"

    # Init
    RUN_NAME = "SwinTransformer_HeavyAugs1"
    EPOCHS = 54

    wandb.init(project="trash_detection_nestiank", entity="bucket_interior", name=RUN_NAME)

    cfg = get_cfg('/opt/ml/detection/swin/configs/heavy_augs/modified_swin_base_heavy_augs.py', RUN_NAME, EPOCHS)

    model = build_detector(cfg.model)
    model.init_weights()

    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.test)]

    # Train
    wandb.alert(title="Train Started", text=f"{RUN_NAME}")
    train_detector(model, datasets[0], cfg, distributed=False, validate=False)
    wandb.alert(title="Train Finished", text=f"{RUN_NAME}")

    # Prediction: Normal Threshold
    checkpoint_path = f"./epoch_{EPOCHS}.pth"

    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])

    data_loader = build_dataloader(
        datasets[1],
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    output = single_gpu_test(model, data_loader, show_score_thr=0.05)

    make_predictions(output, cfg, f"./epoch{EPOCHS}.csv")

    # Prediction: Low Threshold
    cfg = get_cfg('/opt/ml/detection/swin/configs/thr_down/modified_swin_base_heavy_augs_thr_down.py', RUN_NAME, EPOCHS)

    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader, show_score_thr=0.01)

    make_predictions(output, cfg, f"./epoch{EPOCHS}_thr_down.csv")

    wandb.alert(title="Prediction Finished", text=f"{RUN_NAME}")
