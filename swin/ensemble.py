#
# boostcamp AI Tech
# Trash Object Detection Competition
#


from pycocotools.coco import COCO
import pandas as pd
import numpy as np

import argparse

from ensemble_boxes import *


if __name__ == '__main__':
    # Init
    parser = argparse.ArgumentParser()
    parser.add_argument('ensemble', type=str)
    args = parser.parse_args()

    # Ensemble
    assert args.ensemble == 'NMS' or args.ensemble == 'SoftNMS' or args.ensemble == 'WBF', \
        'Ensemble type should be either NMS, SoftNMS or WBF.'

    files = ['./output1.csv', './output2.csv']
    dataframes = [pd.read_csv(file) for file in files]
    image_ids = dataframes[0]['image_id'].tolist()

    annotation = '/opt/ml/detection/dataset/test.json'
    coco = COCO(annotation)

    prediction_strings = []
    file_names = []

    for i, image_id in enumerate(image_ids):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]

        for df in dataframes:
            predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
            predict_list = str(predict_string).split()

            if len(predict_list) <= 1:
                continue

            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                box[0] = float(box[0]) / image_info['width']
                box[1] = float(box[1]) / image_info['height']
                box[2] = float(box[2]) / image_info['width']
                box[3] = float(box[3]) / image_info['height']
                box_list.append(box)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        if len(boxes_list):
            if args.ensemble == 'NMS':
                boxes, scores, labels = nms(boxes_list, scores_list, labels_list)
            elif args.ensemble == 'SoftNMS':
                boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list)
            elif args.ensemble == 'WBF':
                boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list)

            for box, score, label in zip(boxes, scores, labels):
                prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_id)

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(f"./ensemble_{args.ensemble}.csv", index=False)
