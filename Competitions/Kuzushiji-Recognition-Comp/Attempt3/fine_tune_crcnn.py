import tensorflow as tensorflow
import tensorflow_models as tfm
import os

from official.core import exp_factory
from official.core import config_definitions as cfg
from official.vision.data import tfrecord_lib
from official.vision.serving import export_saved_model_lib
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
from official.vision.utils.object_detection import visualization_utils
from official.vision.ops.preprocess_ops import normalize_image, resize_and_crop_image
from official.vision.data.create_coco_tf_record import coco_annotations_to_lists

exp_config = exp_factory.get_exp_config('cascadercnn_spinenet_coco')

train_data_input_path = './data/dtrain_crop.pkl'
val_data_input_path = './data/dval_crop.pkl'
test_data_input_path = './data/dtest.pkl'
model_dir = './model_dir'
export_dir = './export_dir'

# Get model
exp_config = exp_factory.get_exp_config('cascadercnn_spinenet_coco')

# Load the data
batch_size = 32
num_classes = 4781

HEIGHT, WIDTH = 1024, 1024
IMG_SIZE = [HEIGHT, WIDTH, 3]

# Backbone config
exp_config.task.freeze_backbone = True 
exp_config.task.annotation_file = ''

# Model config
exp_config.task.model.input_size = IMG_SIZE
exp_config.task.model.num_classes = num_classes + 1
exp_config.task.model.detection_generator.tflite_post_processing.max_classes_per_detection = exp_config.task.model.num_classes

# Training data config
exp_config.task.train_data.input_path = train_data_input_path
exp_config.task.train_data.dtype = 'float32'
exp_config.task.train_data.global_batch_size = batch_size
exp_config.task.train_data.parser.aug_scale_max = 1.0
exp_config.task.train_data.parser.aug_scale_min = 1.0

# Validation data config.
exp_config.task.validation_data.input_path = val_data_input_path
exp_config.task.validation_data.dtype = 'float32'
exp_config.task.validation_data.global_batch_size = batch_size

# Adjust trainer config
logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]

if 'GPU' in ''.join(logical_device_names):
  print('This may be broken in Colab.')
  device = 'GPU'
elif 'TPU' in ''.join(logical_device_names):
  print('This may be broken in Colab.')
  device = 'TPU'
else:
  print('Running on CPU is slow, so only train for a few steps.')
  device = 'CPU'
