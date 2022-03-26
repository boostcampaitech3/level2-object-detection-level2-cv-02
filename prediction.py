from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmcv.parallel import MMDataParallel
from pycocotools.coco import COCO
import pandas as pd
import argparse

# Init

parser = argparse.ArgumentParser()
parser.add_argument('epoch', type=int)
args = parser.parse_args()

# Prediction

RUN_NAME = "SwinTransformer_DyHead_Epochs60"
checkpoint_path = f"./epoch_{args.epoch}.pth"

cfg = Config.fromfile('/opt/ml/detection/swin/configs/modified_swin_base.py')
cfg.checkpoint_config = dict(max_keep_ckpts=50, interval=2)
cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.log_config.hooks[1].init_kwargs.name = RUN_NAME

cfg.model.neck = [cfg.model.neck, dict(type='DyHead', in_channels=256, out_channels=256)]

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

output = single_gpu_test(model, data_loader, show_score_thr=0.05)

prediction_strings = []
file_names = []

coco = COCO(cfg.data.test.ann_file)
img_ids = coco.getImgIds()

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
submission.to_csv(f"./epoch{args.epoch}.csv", index=None)
