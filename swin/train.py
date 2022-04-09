#
# boostcamp AI Tech
# Trash Object Detection Competition
#


from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import train_detector, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmcv.parallel import MMDataParallel

import wandb

from options import RUN_NAME, CONFIG_PATH, CONFIG_PATH_LOW_THR
from options import get_cfg, make_predictions


if __name__ == '__main__':
    # Init
    BASE_EPOCHS = 0
    EPOCHS = 42 - BASE_EPOCHS

    assert BASE_EPOCHS > EPOCHS, "The original checkpoint will be overwritten."

    wandb.init(project="trash_detection_nestiank", entity="bucket_interior", name=RUN_NAME)

    cfg = get_cfg(CONFIG_PATH, RUN_NAME, EPOCHS)

    model = build_detector(cfg.model)
    if BASE_EPOCHS == 0:
        # Learning from the pretrained weights
        model.init_weights()
    else:
        # Learning from the checkpoint
        assert BASE_EPOCHS > cfg.lr_config.step[1], "Learning rate should be considered carefully."
        cfg.optimizer.lr = 1e-6
        cfg.lr_config = None

        checkpoint_path = f"./epoch_{BASE_EPOCHS}.pth"
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
        model = MMDataParallel(model.cuda(), device_ids=[0])

    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.test)]

    # Train
    wandb.alert(title="Train Started", text=f"{RUN_NAME}")
    train_detector(model, datasets[0], cfg, distributed=False, validate=False)

    # Prediction: Normal Threshold
    checkpoint_path = f"./epoch_{EPOCHS}.pth"

    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])

    data_loader = build_dataloader(
        datasets[1],
        samples_per_gpu=1,
        workers_per_gpu=8,
        dist=False,
        shuffle=False
    )

    output = single_gpu_test(model, data_loader, show_score_thr=0.05)

    make_predictions(output, cfg, f"./epoch{EPOCHS + BASE_EPOCHS}.csv")

    # Prediction: Low Threshold
    cfg = get_cfg(CONFIG_PATH_LOW_THR, RUN_NAME, EPOCHS)

    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader, show_score_thr=0.01)

    make_predictions(output, cfg, f"./epoch{EPOCHS + BASE_EPOCHS}_thr_down.csv")
