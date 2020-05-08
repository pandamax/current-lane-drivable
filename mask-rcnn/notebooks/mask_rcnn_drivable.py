import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, '../libraries')
from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib
from mrcnn.model import log
import mcoco.coco as coco


# HOME_DIR is the path that the project you put on
HOME_DIR = '.../current-lane-drivable-master'
DATA_DIR = os.path.join(HOME_DIR, "data/drivable")
print(DATA_DIR)
WEIGHTS_DIR = os.path.join(HOME_DIR, "data/weights")
print(WEIGHTS_DIR)
MODEL_DIR = os.path.join(DATA_DIR, "logs")
print(MODEL_DIR)

# Local path to trained weights file
DRIVABLE_MODEL_PATH = os.path.join(WEIGHTS_DIR, "mask_rcnn_drivable_res50.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

dataset_train = coco.CocoDataset()
dataset_train.load_coco(DATA_DIR, subset="drivable_train", year="2019")
dataset_train.prepare()

dataset_validate = coco.CocoDataset()
dataset_validate.load_coco(DATA_DIR, subset="drivable_validate", year="2019")
dataset_validate.prepare()

dataset_test = coco.CocoDataset()
dataset_test.load_coco(DATA_DIR, subset="drivable_test", year="2019")
dataset_test.prepare()


# change from coco config
class ShapesConfig(Config):
    """Configuration for training on the shapes dataset.
    """
    NAME = "drivable"

    #choose backbone
    BACKBONE = "resnet50"

    # Train on 1 GPU and 2 images per GPU. Put multiple images on each
    # GPU if the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1(drivable)

    #     # Use smaller images for faster training.
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_DIM = 512
    IMAGE_RESIZE_MODE = "square"

    #     # Use smaller anchors because our image and objects are small
    #     RPN_ANCHOR_SCALES = rpn_anchor_scales

    #     # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    STEPS_PER_EPOCH = 1000

    VALIDATION_STEPS = 50
    POST_NMS_ROIS_INFERENCE = 512

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

config = ShapesConfig()
config.display()

model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# inititalize_weights_with = "coco"  # imagenet, coco, or last
# if inititalize_weights_with == "imagenet":
#     model.load_weights(model.get_imagenet_weights(), by_name=True)
#
# elif inititalize_weights_with == "coco":
#     model.load_weights(COCO_MODEL_PATH, by_name=True,
#                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
#                                 "mrcnn_bbox", "mrcnn_mask"])
#
# elif inititalize_weights_with == "last":
# Load the last model you trained and continue training
# model.load_weights(model.find_last()[1], by_name=True)

model.load_weights(DRIVABLE_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_train, dataset_validate,
            learning_rate=config.LEARNING_RATE,
            epochs=4,
            layers='heads')
model.train(dataset_train, dataset_validate,
            learning_rate=config.LEARNING_RATE,
            epochs=12,
            layers='4+')

model.train(dataset_train, dataset_validate,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=16,  # starts from the previous epoch, so only 1 additional is trained
            layers="all")

######################################
#            Add augmentation        #
######################################
#  # Training - Stage 1
# print("Training network heads")
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=40,
#             layers='heads',
#             augmentation=augmentation)
#
# # Training - Stage 2
# # Finetune layers from ResNet stage 4 and up
# print("Fine tune Resnet stage 4 and up")
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=120,
#             layers='4+',
#             augmentation=augmentation)
#
# # Training - Stage 3
# # Fine tune all layers
# print("Fine tune all layers")
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE / 10,
#             epochs=160,
#             layers='all',
#             augmentation=augmentation)