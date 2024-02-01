import random
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

import mmcv
from mmdet.evaluation.functional import bbox_overlaps
from mmengine import load, dump

BASE_DIR = '/home/harris/Projects/ML/Datasets/Kuzushiji-Recognition'

def np_iof(bboxes, canvas):
    ''' Gets intersection of foreground, ie 
    % of area shared between two bounding boxes '''
    return bbox_overlaps(
        bboxes,
        canvas,
        mode='iof'
        ).reshape(-1)

def iter_empty_crops(w, h, stride, size):
    ''' Generator yielding all possible crops of size `size`'''

    # Get all possible x positions
    xs = range(0, w - size + 1, stride)
    xs = sorted(set(list(xs) + [w - size]))
    
    # Get all possible y positions
    ys = range(0, h - size + 1, stride)
    ys = sorted(set(list(ys) + [h - size]))

    for x in xs:
        for y in ys:
            yield [x, y, x + size, y + size]

def iter_target_crops(bboxes, w, h, size):
    available_bboxes = bboxes.copy()
    while len(available_bboxes) > 0: 
        # randomly pick a bbox
        idx = np.random.choice(range(len(available_bboxes)))
        bbox = available_bboxes[idx]

        x1, y1, x2, y2 = bbox
        lt_region = np.array([x2 - size, y2 - size, x1, y1])
        lt_region[0::2] = lt_region[0::2].clip(0, w-size)
        lt_region[1::2] = lt_region[1::2].clip(0, h-size)

        assert(lt_region[2] >= lt_region[0] and lt_region[3] >= lt_region[1])

        # make a crop
        x = np.random.randint(lt_region[0], lt_region[2] + 1)
        y = np.random.randint(lt_region[1], lt_region[3] + 1)
        crop = np.array([[x, y, x + size, y + size]])

        # Check if crop is valid
        inds = np_iof(available_bboxes, crop) > 0.8
        available_bboxes = available_bboxes[~inds] 
        yield crop.flatten().tolist()

def crop_gt(bboxes, labels, x1, y1, x2, y2):
    crop = np.array([[x1, y1, x2, y2]])
    iof = np_iof(bboxes, crop)
    inds = iof > 0.8
    ret_labels = labels[inds].copy()
    ret_bboxes = bboxes[inds].copy()
    ret_bboxes -= np.array([x1, y1, x1, y1])
    ret_bboxes[:, 0::2] = ret_bboxes[:, 0::2].clip(0, x2 - x1)
    ret_bboxes[:, 1::2] = ret_bboxes[:, 1::2].clip(0, y2 - y1)
    return ret_bboxes, ret_labels


def main(data_loc):
    random.seed(0)
    np.random.seed(0)

    # Load all train_data   
    data = load(Path('data', data_loc))
    d_crop = []

    # Define crop size & dummy label
    SIZE = 1024
    LABEL_DUMMY = 4782

    crops = []
    for sample in tqdm(data):
        img = mmcv.imread(Path(BASE_DIR, 'train_images' + '/' + sample['filename']))
        
        # Get manuiscript dimensions & char dimensions
        w, h = sample['width'], sample['height']
        bboxes = sample['ann']['bboxes']
        labels = sample['ann']['labels']
        idx_crop = 0
        base_name = sample['filename'].rstrip('.jpg')

        if len(bboxes) == 0:
            for x1, y1, x2, y2 in iter_empty_crops(w, h, SIZE - 64, SIZE):
           
                img_crop = img[y1:y2, x1:x2]
                # ???????
                img_crop[8:24, 8:24] = img_crop[8:24, 8:24] * 0.5  + [0, 127, 0]
                bboxes_crop = np.array([[8, 8, 24, 24]], np.float32)
                labels_crop = np.array([LABEL_DUMMY], np.int64)

                filename = f'{base_name}_{idx_crop}.jpg'

                mmcv.imwrite(
                    img_crop,
                    Path(BASE_DIR, 'train_images_crop') + '/' + filename,
                    auto_mkdir=True
                )

                d_crop.append({
                    'filename': filename,
                    'width': SIZE,
                    'height': SIZE,
                    'ann': {
                        'bboxes': bboxes_crop.reshape(-1, 4),
                        'labels': labels_crop.reshape(-1),
                        'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
                        'labels_ignore': np.array([], dtype=np.int64).reshape(-1)
                    },
                })
                idx_crop += 1
        else:
            for x1, y1, x2, y2 in iter_target_crops(bboxes, w, h, SIZE):
                img_crop = img [y1:y2, x1:x2]
                bboxes_crop, labels_crop = crop_gt(bboxes, labels, x1, y1, x2, y2)

                filename = f'{base_name}_{idx_crop}.jpg'
                mmcv.imwrite(
                    img_crop,
                    Path(BASE_DIR, 'train_images_crop' + '/' + filename),
                    auto_mkdir=True
                )

                d_crop.append({
                    'filename': filename,
                    'width': SIZE,
                    'height': SIZE,
                    'ann': {
                        'bboxes': bboxes_crop.reshape(-1, 4),
                        'labels': labels_crop.reshape(-1),
                        'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
                        'labels_ignore': np.array([], dtype=np.int64).reshape(-1)
                    },
                })
                idx_crop += 1
        
    dump(d_crop, Path('data', data_loc[:-4] + '_crop.pkl'))

if __name__ == "__main__":
    main('dtrain.pkl')
    main('dval.pkl')