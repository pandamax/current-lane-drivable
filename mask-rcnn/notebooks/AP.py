import os
import sys
sys.path.insert(0, '../libraries')
import mextra.utils as extra_utils
import numpy as np
import mcoco.coco as coco
import mrcnn.model as modellib
from mrcnn.config import Config

# HOME_DIR is the path that the project you put on
HOME_DIR = '.../current-lane-drivable-master'
MODEL_DIR = os.path.join(HOME_DIR, "data/weights")
# prepare data to test
DATA_DIR = os.path.join(HOME_DIR, "data/drivable")
dataset_test = coco.CocoDataset()
dataset_test.load_coco(DATA_DIR, subset="drivable_test", year="2019")
dataset_test.prepare()

# inference config
class InferenceConfig(Config):
    NAME = "drivable"

    # Train on 1 GPU and 2 images per GPU. Put multiple images on each
    # GPU if the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1(drivable)

    #     # Use smaller images for faster training.
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_DIM = 800
    IMAGE_RESIZE_MODE = "square"

    #     # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    STEPS_PER_EPOCH = 1000

    VALIDATION_STEPS = 50

    BACKBONE = "resnet101"
    # if you want to speed up the inference time, Reduce input size!

    POST_NMS_ROIS_INFERENCE = 512


config = InferenceConfig()
config.display()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=config,
                          model_dir=MODEL_DIR)

# Load weights for test
weights_path = os.path.join(MODEL_DIR, 'mask_rcnn_drivable_res101.h5')
print(weights_path)

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

print("---------just wait, keep patient~--------------")

start = timen=
predictions =\
extra_utils.compute_multiple_per_class_precision(model, config, dataset_test,
                                                 number_of_images=500, iou_threshold=0.5)
complete_predictions = []

for drivable in predictions:
    complete_predictions += predictions[drivable]
    print("{} ({}): {}".format(drivable, len(predictions[drivable]), np.mean(predictions[drivable])))

print("---------------------------------------------------------------------------------")
print("average precision: {}".format(np.mean(complete_predictions)))