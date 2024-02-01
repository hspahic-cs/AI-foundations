import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from mmengine import load, dump

import numpy as np
import random

BASE_DIR = '/home/harris/Projects/ML/Datasets/Kuzushiji-Recognition'

def iter_boxes(labels):
    '''
    Generator getting bounding boxes for each image

    (list) labels -> a list of the particular page's labels
    '''

    if not labels:
        return
    
    # Comes in 5 tuples: char, x, y, w, h
    labels = labels.split()
    n = len(labels)
    assert n % 5 == 0
    
    for i in range(0, n, 5):
        ch, x, y, w, h = labels[i:i+5]
        yield ch, int(x), int(y), int(w), int(h)

def prepare_train():
    # Read in the data
    df = pd.read_csv(Path(BASE_DIR, 'train.csv'), keep_default_na=False)
    img_dir = Path(BASE_DIR, 'train_images')

    # df = df[0:10]

    # Get unicode translations
    unicode_translation = pd.read_csv(Path(BASE_DIR, 'unicode_translation.csv'))
    unicode_conversion = dict(zip(unicode_translation['Unicode'], unicode_translation.index.values))

    # Add images to COCO
    images = []
    for idx, row in tqdm(df.iterrows()):
        filename = row['image_id'] + '.jpg'
        filepath = Path(img_dir, filename)
        img = Image.open(filepath)

        image = {
            'filename': filename,
            'width': img.width,
            'height': img.height
        }

        bboxes = []
        labels = []

        for ch, x, y, w, h in iter_boxes(row['labels']):
            bboxes.append([x, y, w+x, h+y])
            labels.append(4782) if ch not in unicode_conversion.keys() else labels.append(unicode_conversion[ch] + 1)

        image['ann'] = {
            'bboxes': np.array(bboxes).astype(np.float32).reshape(-1, 4),
            'labels': np.array(labels).astype(np.int64).reshape(-1),
            'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
            'labels_ignore': np.array([], dtype=np.int64).reshape(-1)
        }


        images.append(image)

    # Shuffle the images
    random.shuffle(images)
    split = int(len(images) * 0.8)

    dump(images[:split], 'data/dtrain.pkl')        
    dump(images[split:], 'data/dval.pkl')
    dump(images, 'data/trainval.pkl')
    
def prepare_test():
    df = pd.read_csv(Path(BASE_DIR, 'sample_submission.csv'), keep_default_na=False)
    img_dir = Path(BASE_DIR, 'test_images')

    # df = df[0:10]

    # Read in more images
    images = []
    for idx, row in tqdm(df.iterrows()):
        filename = row['image_id'] + '.jpg'
        filepath = Path(img_dir, filename)
        img = Image.open(filepath)

        images.append({
            'filename': filename,
            'width': img.width,
            'height': img.height,
            'ann': {
                'bboxes': np.array([]).astype(np.float32).reshape(-1, 4),
                'labels': np.array([]).astype(np.int64).reshape(-1),
                'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array([], dtype=np.int64).reshape(-1)
                }
            }
        )
    
    dump(images, 'data/dtest.pkl')

if __name__ == "__main__":
    prepare_train()
    prepare_test()  