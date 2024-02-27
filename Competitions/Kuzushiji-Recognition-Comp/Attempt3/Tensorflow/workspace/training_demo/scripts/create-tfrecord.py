import tensorflow as tf
import io
from PIL import Image
from pathlib import Path
import numpy as np

# Step 0: Define constants
BASE_DIR = '/home/harris/Projects/ML/Datasets/Kuzushiji-Recognition'

# Step 1: Prep for byte conversion
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# Step 2: Read in the data
def read_coco(record):
    # Convert data to bytes
    img = Image.open(Path(BASE_DIR, record['filename']))
    image_np = np.array(img)
    image_data = io.BytesIO()

    bboxes = record['bboxes']
    labels = record['labels']
    
    feature_map = {
        'image/height': _int64_feature(record['height']),
        'image/width': _int64_feature(record['width']),
        'image/data': _int64_feature(img),
        'image/bboxes': _int64_feature(bboxes),
        'image/labels': _int64_feature(labels)
    }

    # Create feature message using tf.train.Example
    example = tf.train.Examples(features=tf.train.Features(feature=feature_map)) 

    # Serialize to a string
    serealized_example = example.SerializeToString()

    return serealized_example