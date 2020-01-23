# -*- coding: utf-8 -*-
"""Detectron2 
* Run inference on images or videos, with an existing detectron2 model
* Train a detectron2 model on a new dataset
"""

import torch, torchvision

# You may need to restart your runtime prior to this, to let your installation take effect

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import sys

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

"""# Train on a daifuku dataset

In this section, we show how to train an existing detectron2 model on a custom dataset in a new format.

We'll train a segmentation model from an existing model pre-trained on COCO dataset, available in detectron2's model zoo.

## Prepare the dataset
"""

"""Register the daifuku dataset to detectron2, following the [detectron2 custom dataset tutorial](https://detectron2.readthedocs.io/tutorials/datasets.html).
Here, the dataset is in its custom format, therefore we write a function to parse it and prepare it into detectron2's standard format. See the tutorial for more details.
"""

import os
import numpy as np
import json
from detectron2.structures import BoxMode

from detectron2.data import DatasetCatalog, MetadataCatalog



# for d in ["train", "val"]:
for d in ["train_coco","val_coco"]:    
    dicts = detectron2.data.datasets.load_coco_json("melbourne/" +d + "/annotations.json", "melbourne/" + d, dataset_name=None, extra_annotation_keys=None)
    DatasetCatalog.register("melbourne_" + d, lambda d=d: dicts)
    MetadataCatalog.get("melbourne_" + d).set(thing_classes=['_background_', 'suitcase','soft_bag','wheel','extended_handle','person','tray','upright_suitcase','spilled_bag','sphere_bag',
'documents','bag_tag','strap_around_bag','stroller','golf_bag','surf_equipment','sport_equipment','music_equipment',
'plastic_bag','shopping_bag','wrapped_bag','umbrella','storage_container','box','big_wheel','laptop_bag','tube','pet_container',
'ski_equipment','tripod','child_safety_car_seat','tool_box','very_small_parcel','bingo_sticker'])
melbourne_metadata = MetadataCatalog.get("melbourne_train_coco") # or melbourne_val_coco


"""To verify the data loading is correct, let's visualize the annotations of randomly selected samples in the training set:"""

dataset_dicts = detectron2.data.datasets.load_coco_json("melbourne/train_coco/annotations.json", "melbourne/train_coco", dataset_name=None, extra_annotation_keys=None)

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=melbourne_metadata, scale=0.5)
    # print(d)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("data loaded", vis.get_image()[:, :, ::-1])
    cv2.waitKey(1000)
    # cv2.destroyAllWindows() 

"""## Train!

Now, let's fine-tune a coco-pretrained R50-FPN Mask R-CNN model on the melbourne dataset. It takes ~6 minutes to train 300 iterations on Colab's K80 GPU, or ~2 minutes on a P100 GPU.
"""

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("melbourne_train_coco",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl" 
# cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.SOLVER.IMS_PER_BATCH = 2 # 16
cfg.SOLVER.BASE_LR = 0.02 #0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 (15000: 9hours) iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8192   # 128 faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 33  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 

import timeit
start = timeit.default_timer()
print("Model loading time:,", start)
trainer.resume_or_load(resume=False)
stop = timeit.default_timer()
print("loading time:,", stop - start)

start = timeit.default_timer()
trainer.train()
stop = timeit.default_timer()
print('........   training finshed. Time: ', stop - start)  


# Commented out IPython magic to ensure Python compatibility.
# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output

"""## Inference & evaluation using the trained model
Now, let's run inference with the trained model on the melbourne validation dataset. First, let's create a predictor using the model we just trained:
"""

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("melbourne_val_coco", )
predictor = DefaultPredictor(cfg)

"""Then, we randomly select several samples to visualize the prediction results."""

from detectron2.utils.visualizer import ColorMode
json_file = ("melbourne/val_coco/annotations.json")
image_root = ("melbourne/val_coco")
dataset_dicts = detectron2.data.datasets.load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None)

for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=melbourne_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("predicted result", v.get_image()[:, :, ::-1])
    cv2.waitKey(1000)
    # cv2.destroyAllWindows()

"""We can also evaluate its performance using AP metric implemented in COCO API.
This gives an AP of ~70%. Not bad!
"""

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("melbourne_val_coco", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "melbourne_val_coco")
inference_on_dataset(trainer.model, val_loader, evaluator)
# another equivalent way is to use trainer.test

