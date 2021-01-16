import skimage.io as io
import os 
import pandas as pd
from skimage.color import rgb2gray
import numpy as np
from skimage.transform import resize


def load_dataset(path_to_dataset):
    features = []
    labels = []
    for x in range(0,len(path_to_dataset)):
        img_filenames = os.listdir(path_to_dataset[x])

        for i, fn in enumerate(img_filenames):
            if fn.split('.')[-1] != 'bmp' and fn.split('.')[-1] != 'jpg':
                continue

            label = fn.split('.')[0]
            labels.append(label)

            path = os.path.join(path_to_dataset[x], fn)
            img = rgb2gray(io.imread(path))
            img = np.where(img > 0.5, 1, 0)
            features.append(extract_segments_ones(img))

    df = pd.DataFrame(features)
    df['labels']=labels
    df.to_csv('featuresAndLabels.csv')
    return


def extract_segments_ones(img, target_img_size = (64,64)):
    imgCopied = resize(img, target_img_size)
    mx = np.amax(imgCopied)
    imgCopied = imgCopied / mx
    imgCopied = np.where(imgCopied > 0.5, 1, 0)
    arr = list()
    for i in range(0, 4):
        arr.append(np.sum(imgCopied[i * 8:(i + 1) * 8, i * 8:(i + 1) *8 ] == 1)/64)
    return arr