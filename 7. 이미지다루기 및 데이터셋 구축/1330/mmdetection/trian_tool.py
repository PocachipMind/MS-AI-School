# mmdetection 폴더 안의 코드 생성된것.
# https://velog.io/@dust_potato/MM-Detection-Config-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-3
import mmcv
import torch

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import set_random_seed
from mmdet.apis import train_detector
from mmdet.utils import collect_env, get_root_logger, setup_multi_processes
from mmcv.runner import get_dist_info, init_dist
from mmcv import Config

# Dataset register # 이정보들이 어디서 왔냐면 mmdet/datasets/coco.py 에서 왔습니다.
@DATASETS.register_module(force=True) # True를 통해 변경
class WineLabelsDataset(CocoDataset) :
    CLASSES = ('wine-labels', 'AlcoholPercentage',
               "Appellation AOC DOC AVARegion",
               "Appellation QualityLevel",
               "CountryCountry",
               "Distinct Logo",
               "Established YearYear",
               "Maker-Name",
               "Organic",
               "Sustainable",
               "Sweetness-Brut-SecSweetness-Brut-Sec",
               "TypeWine Type",
               "VintageYear",
               )

# config # 모델의 아키텍쳐가 있는 곳 # mask_rcnn은 세그멘테이션 전용 모델입니다.
config_file = "./configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py" # mask 들은 세그멘테이션 관련, 나머지는 디텍션 관련 모델
cfg = Config.fromfile(config_file)
# print(cfg.pretty_text)

# Learning rate setting
# sing GPU -> 0.0025
# cfg.optimizer.lr = 0.02/8
cfg.optimizer.lr = 0.0025 # _base_/schedules 가면 있는거 

# dataset setting # _base_/datasets/coco_detection.py 수정해준거
cfg.dataset_type = "WineLabelsDataset"
cfg.data_root = "./dataset"

# train, val test dataset >> type data root and file img_prefix setting
# train
cfg.data.train.type = "WineLabelsDataset" # 타입은 만든 클래스 이름과 동일해야함
cfg.data.train.ann_file = "./dataset/train/_annotations.coco.json"
cfg.data.train.img_prefix = "./dataset/train/"

# val
cfg.data.val.type = "WineLabelsDataset"
cfg.data.val.ann_file = "./dataset/valid/_annotations.coco.json"
cfg.data.val.img_prefix = "./dataset/valid/"

# test
cfg.data.test.type = "WineLabelsDataset"
cfg.data.test.ann_file = "./dataset/test/_annotations.coco.json"
cfg.data.test.img_prefix = "./dataset/test/"

# class number
cfg.model.roi_head.bbox_head.num_classes = 13

# small obj 잡기위해 -> change anchor -> df : size 8 -> size 4
cfg.model.rpn_head.anchor_generator.scales = [4]

# pretrained call
# open-mmlab/mmdetection/tree/configs/dynamic_rcnn의 Readme에서 다운을 해야함. ( https://download.openmmlab.com/mmdetection/v2.0/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x/dynamic_rcnn_r50_fpn_1x-62a3f276.pth )
cfg.load_from = "./dynamic_rcnn_r50_fpn_1x-62a3f276.pth"

# train model save dir
cfg.work_dir = "./work_dirs/0130"

# lr hyp setting
# configs/_base_/schedules 에 있는거 수정하는거임
cfg.lr_config.warmup = None
cfg.log_config.interval = 10 # 로그 띄우기가 10번째 마다

# 평가 모드로 변환 하는 것임.
# cocodataset evaluation type = bbox
# mAP iou threshold 0.5 ~ 0.95
cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 10 # 몇 번 마다 볼것인지.
cfg.checkpoint_config.interval = 10 # 몇 번 마다 저장할 것인지.

# epoch setting
# 8 x 12 -> 96 # _base_/schedules 가보면 max_epochs가 12인데 GPU가 8개이므로 그런거임. 우리는 바꿔줘야함 
cfg.runner.max_epochs = 96
cfg.seed = 777 # 아무거나
cfg.data.samples_per_gpu = 6 # 배치사이즈를 의미하는건데 요 두개는 고정하면 되요. 더 높이거나 낮추기 돌리기어려움
cfg.data.workers_per_gpu = 2 
print("cfg.data >>" , cfg.data)
cfg.gpu_ids = range(1) # 지피유 갯수 몇개인지 파악
cfg.device = "cuda"
set_random_seed(777, deterministic=False)
print("cfg info show ", cfg.pretty_text)

datasets = [build_dataset(cfg.data.train)]
print("dataset[0]", datasets[0])

# datasets[0].__dict__ variables key val
datasets[0].__dict__.keys()

model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"),
                       test_cfg=cfg.get('test_cfg'))
model.CLASEES = datasets[0].CLASSES
print(model.CLASEES)

# 학습돌리기
if __name__ == '__main__' :
    train_detector(model, datasets,cfg,distributed=False, validate=True)