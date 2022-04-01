#
# boostcamp AI Tech
# Trash Object Detection Competition
#


from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmcv.parallel import MMDataParallel

import argparse

from options import RUN_NAME, CONFIG_PATH, CONFIG_PATH_LOW_THR
from options import get_cfg, make_predictions


if __name__ == '__main__':
    # Init
    parser = argparse.ArgumentParser()
    parser.add_argument('epoch', type=int)
    parser.add_argument('thr_down', default=False, type=bool)
    args = parser.parse_args()

    # Prediction
    checkpoint_path = f"./epoch_{args.epoch}.pth"

    if args.thr_down:
        cfg = get_cfg(CONFIG_PATH_LOW_THR, RUN_NAME, args.epoch)
    else:
        cfg = get_cfg(CONFIG_PATH, RUN_NAME, args.epoch)

    model = build_detector(cfg.model)

    dataset = build_dataset(cfg.data.test)

    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    model = MMDataParallel(model.cuda(), device_ids=[0])

    if args.thr_down:
        output = single_gpu_test(model, data_loader, show_score_thr=0.01)
    else:
        output = single_gpu_test(model, data_loader, show_score_thr=0.05)

    make_predictions(output, cfg, f"./epoch{args.epoch}.csv")
