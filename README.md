# Trash Object Detection Competition

> [boostcamp AI Tech](https://boostcamp.connect.or.kr) - Level 2: CV_02 Bucket Interior

### Results

  * **Test dataset for public leaderboard**
    * mAP score: 0.6279
  * **Test dataset for private leaderboard**
    * mAP score: 0.6057

### Task

#### Object Detection Task Specifications

  * **주어진 쓰레기 사진에서 10종류의 쓰레기 영역을 검출**
    * **Subtask 1: 개별 쓰레기 영역 검출**
      * Bounding box suggestion
    * **Subtask 2: 검출된 영역의 쓰레기 분류**
      * Trashes in bounding box classification

#### Image Dataset Specifications

  * Dataset ratio
    * Train & validation dataset: 50.06%
    * Test dataset for public leaderboard: 24.97%
    * Test dataset for private leaderboard: 49.94%
      * Public test dataset: 24.97%
      * Private test dataset: 24.97%

#### Main Difficulties

  * Data imbalance
    * 일부 유형의 쓰레기 사진의 비율이 유의미하게 낮았음
  * Bounding box noise
    * 많은 쓰레기 사진의 쓰레기 영역 annotation이 정확하지 않았음
  * Tiny objects
    * 대부분의 쓰레기 사진에 등장하는 쓰레기의 크기가 매우 작았음
    * 매우 많은 숫자의 매우 작은 쓰레기가 등장하는 사진이 유의미하게 많았음

### Approaches

  * Selecting models
    * 코드가 공개되어 있는 SOTA 모델부터 순서대로 실험에 사용
    * Ensemble 과정에서의 다양성을 위해 여러 모델을 실험에 사용
      * Swin Transformer(Swin-B)
      * UniverseNet(101)
      * YOLO(v3, v5)
      * EfficientDet(b3, b5, b7)
      * RetinaNet
      * Faster R-CNN
  * Dealing with tiny objects
    * 다양한 heavy augmentation을 실험에 사용
      * RandomCrop
      * RandomFlip
      * PhotoMetricDistortion
      * RandomFog
      * Corrupt: Gaussian Noise
      * CutOut
  * Increasing generalization performance
    * 다양한 learning rate를 실험에 사용
      * 시작: 1e-2, 5e-3, 1e-3, 5e-4, 1e-4
      * 관리: fixed, cosine annealing, step decay
    * 다양한 optimizer를 실험에 사용: SGD, Adam, AdamW
  * Score-related tweaks
    * Prediction with low thresholds
    * 다양한 ensemble 방법을 실험에 사용: NMS, Soft-NMS, WBF

### Technical Specifications

> 가장 높은 mAP score를 달성한 모델에 대해서만 기록

  * Model: WBF ensemble
    * Swin-B
    * UniverseNet-101
  * Train & validation dataset split rule
    * 최종 답안은 validation dataset 없이 학습한 모델로 생성

### Thoughts

> 이번 프로젝트에서도, 우리는 [직전 프로젝트][image-classification-project]의 경험을 살려, EDA, data augmentation, modeling 등을 빠짐 없이 차례대로 수행해 나갔고, 다양한 모델을 실험에 사용하였으며, ensemble도 적용해 보았다. 그리고 이번에는 WandB나 FiftyOne과 같은 유용한 도구들을 더 많이 사용하여 프로젝트를 진행하는 등, 직전 프로젝트보다 개선된 프로젝트 pipeline을 확보하였다. 이것은 직전 프로젝트를 마치고 나서 논의한 개선 방향을 실천에 옮긴 것으로, 충분히 칭찬할 만 하다. <br><br>
> 프로젝트 기간이 더욱 길었더라면 보다 일관성 있는 실험 계획을 할 수 있었을 것이라는 점이 아쉽다. 그리고 우수 사례 발표를 통해 모델 크기의 중요성을 깊이 체감했다. Swin transformer를 사용하기로 결정한 것은 잘한 일이지만, object detection task를 위한 pretrained weight의 부재로 인해 Swin-L 대신 Swin-B를 사용한 점은 성능 향상에 한계가 생기는 원인이 되었다. 실제로 상위권 팀들은 ImageNet 기반 pretrained weight을 가져와서 Swin-L을 사용하였다고 하였기에, 만일 우리도 Swin-L을 사용하였다면 성능이 훨씬 좋았을 것으로 생각된다. 예정되어 있는 semantic segmentation 프로젝트에서는 Swin-B를 사용하지 않기로 하였다. <br><br>
> 이번 프로젝트는 여기에 적은 점들 이외에도 많은 교훈을 주었다. 그리고 그런 교훈들을 자연스럽게 인지하고 어느새 다음 프로젝트를 위한 개선점을 찾고 있는 우리의 모습을 보며, 다음 프로젝트에서는 훨씬 좋은 성과가 있으리라는 믿음을 가지게 되었다. 따라서 이번 프로젝트도 성공적으로 마무리한 것이다.

<!-- Link Definition -->
[image-classification-project]: https://github.com/boostcampaitech3/level1-image-classification-level1-cv-06
