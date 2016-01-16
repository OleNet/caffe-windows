import h5py
import numpy as np
import os
import skimage.io as skio

ROOT = 'D:/Project/caffe-windows-master/data/Blur1500/'


def load_h5(h5_path):
    data = h5py.File(h5_path, 'r')
    keys = data.keys()
    return data, keys


def test1():
    data, key = load_h5(ROOT + os.sep + 'train320.h5')
    X = data['X']
    img = np.squeeze(X[700])
    score = data['y'][700]
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    skio.imshow(img)
    print score
    skio.show()


if __name__ == '__main__':
    test1()