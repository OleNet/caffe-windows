# -*-coding=utf-8-*-

import h5py, os
import caffe
import numpy as np

ROOT = 'D:/Project/caffe-windows-master/data/Blur1000/'
PHASE = 'test'
SIZE = 170 # fixed size to all images
HD5SIZE = 100
INI_VALUE = -1000

def load_caffe_format_img(img_path):
    img = caffe.io.load_image(img_path)
    img = caffe.io.resize( img, (SIZE, SIZE, 3) ) # resize to fixed size
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    return img


def allocate_XY(num, image_size):
    # If you do not have enough memory split data into
    # multiple batches and generate multiple separate h5 files
    X = np.zeros((num, 3, image_size[0], image_size[1]), dtype='f4') + INI_VALUE
    y = np.zeros((1, num), dtype='f4' ) + INI_VALUE
    return X, y


X, y = allocate_XY(HD5SIZE, (SIZE, SIZE))
with open( ROOT + os.sep + 'test.txt', 'r' ) as T :
    lines = T.readlines()
valid_count = 0
for i, l in enumerate(lines):
    sp = l.split(' ')
    img_path = ROOT + os.sep + sp[0]
    if not os.path.exists(img_path):
        print 'file not exist {}'.format(img_path)
        continue;

    img = load_caffe_format_img(img_path)
    # you may apply other input transformations here...
    X[valid_count % HD5SIZE ] = img
    y[0, valid_count % HD5SIZE ] = float(sp[1])
    valid_count += 1

    if valid_count % HD5SIZE == 0:
        valid_ind = (y != INI_VALUE)  # 避免对X，Y填充不满的情况
        with h5py.File(ROOT + os.sep + PHASE + str(valid_count) + '.h5','w') as H:
            H.create_dataset('X', data=X[np.squeeze(valid_ind)]) # note the name X given to the dataset!
            H.create_dataset('y', data=y[valid_ind]) # note the name y given to the dataset!

        with open(ROOT + os.sep + PHASE + '_h5_list.txt', 'a') as L:
            L.write(ROOT + os.sep + PHASE + str(valid_count) + '.h5\n') # list all h5 files you are going to use
        X, y = allocate_XY(HD5SIZE, (SIZE, SIZE))

valid_ind = (y != INI_VALUE)
with h5py.File(ROOT + os.sep + PHASE + str(valid_count) + '.h5','w') as H:
    H.create_dataset('X', data=X[np.squeeze(valid_ind)]) # note the name X given to the dataset!
    H.create_dataset('y', data=y[valid_ind]) # note the name y given to the dataset!
with open(ROOT + os.sep + PHASE + '_h5_list.txt', 'a') as L:
    L.write(ROOT + os.sep + PHASE + str(valid_count) + '.h5') # list all h5 files you are going to use