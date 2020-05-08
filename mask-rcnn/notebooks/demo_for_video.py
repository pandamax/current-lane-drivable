from keras.preprocessing import image
import os
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '../libraries')
from mrcnn.config import Config
import mrcnn.model as modellib
import cv2
from moviepy.editor import VideoFileClip
from mrcnn import visualize_drivable

# HOME_DIR is the path that the project you put on
HOME_DIR = '/home/pandamax/current-lane-drivable-master'
DATA_DIR = os.path.join(HOME_DIR, "data/drivable")
WEIGHTS_DIR = os.path.join(HOME_DIR, "data/weights")
MODEL_DIR = os.path.join(DATA_DIR, "logs")


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

    #################################
    # Set your config for inference #
    #################################
class InferenceConfig(Config):
    """Configuration for training on the shapes dataset.
    """
    NAME = "drivable"

    # Choose BACKBONE from ["resnet50","resnet101"]
    BACKBONE = "resnet101"

    # Train on 1 GPU and 2 images per GPU. Put multiple images on each
    # GPU if the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1+1  # background + 1(drivable)

    # Use smaller images for faster inference
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_DIM = 800
    IMAGE_RESIZE_MODE = "square"

    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    POST_NMS_ROIS_INFERENCE = 1000

# show inference config
inference_config = InferenceConfig()
inference_config.display()
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")

# print(model.find_last()[1])
# model_path = model.find_last()[1]

# choose the trained model to run inference stage
model_path = os.path.join(WEIGHTS_DIR, "mask_rcnn_drivable_res101.h5")
print("The Model Is Loading ...")

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
model.load_weights(model_path, by_name=True)

# two class
class_names = ['BG', 'drivable']

def process_video(image, title="", figsize=(16, 16), ax=None):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes
    results = model.detect([image], verbose=0)
    r = results[0]
    image = visualize_drivable.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                       class_names, r['scores'])
    return image


output = os.path.join(HOME_DIR, "mask-rcnn/notebooks/test_out/harder_challenge_video_1280 x 720.mp4")
clip1 = VideoFileClip(os.path.join(HOME_DIR, "mask-rcnn/notebooks/test_video/harder_challenge_video_1280 x 720.mp4"))
clip = clip1.fl_image(process_video) #NOTE: this function expects color images!!
clip.write_videofile(output, audio=False)

print("Process Successfully!")
