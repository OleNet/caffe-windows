# -*-coding=utf-8-*-

import h5py, os
import caffe
import numpy as np
import skimage.io as skio
import img_mani
import label_file_util as label_util
import random

ROOT = 'D:/Project/caffe-windows-master/data/Blur1500/'
PHASE = 'test'
LABEL_FILE = 'D:/Project/caffe-windows-master/data/Blur1500/' + PHASE + '.proto'

SIZE = 170 # fixed size to all images
HD5SIZE = 800
INI_VALUE = -1000


def swap_caffe_axies(narray):
    narray = np.swapaxes(narray, 0, 2)
    narray = np.swapaxes(narray, 1, 2)
    return narray


def caffe_format_img(img):
    img = caffe.io.resize(img, (SIZE, SIZE, 3)) # resize to fixed size
    img = swap_caffe_axies(img)
    return img


def load_image(img_path):
    if not os.path.exists(img_path):
        print '\tfile not exist {}'.format(img_path)
        return False
    img = caffe.io.load_image(img_path)
    return img


def load_caffe_format_img(img_path):
    img = load_image(img_path)
    return caffe_format_img(img)


def write_XY_h5(h5name, X, y, shuffle=False):
    if shuffle:
        randidx = np.random.permutation(X.shape[0])
        X = X[randidx]
        y = y[randidx]
    with h5py.File(h5name, 'w') as H:
        H.create_dataset('X', data=X)
        H.create_dataset('y', data=y)


def resize_batch(imgs):
    for i, img in enumerate(imgs):
        imgs[i] = caffe.io.resize(img, (SIZE, SIZE, 3))
    return imgs


def augment_img(img):
    #augment here
    img_augs = img_mani.augment_image(img)
    img_augs.append(img)
    img_augs = resize_batch(img_augs)
    img_augs = np.stack(img_augs)
    return img_augs


def get_empty_XY():
    X = np.empty(shape=(0, 3, SIZE, SIZE))
    y = np.empty(shape = 0)
    return X, y


def write_h5names(filename, h5name_list):
    with open(filename, 'w') as L:
        for name in h5name_list:
            L.write(name + '\n' ) # list all h5 files you are going to use


def save_batch(name_seed, imgs, additional_name_info=[]):
    temp_dir = './_temp/'
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    name, _ = os.path.splitext(name_seed)
    for i, img in enumerate(imgs):
        skio.imsave(temp_dir + os.sep + additional_name_info + '-{}-{}.bmp'.format(name, i), img)


def shuffleL1L2(a, b):
    combined = zip(a, b)
    random.shuffle(combined)

    a[:], b[:] = zip(*combined)
    return a, b

def convert2h5data():
    Xs, ys = get_empty_XY()
    count = 0
    h5name_record = []
    '''if os.path.splitext(LABEL_FILE)[1] == '.txt':
        file_list, score_list = label_util.load_old_format_labelscore(LABEL_FILE)
    if os.path.splitext(LABEL_FILE)[1] == '.proto':
        file_list, score_list = label_util.get_name_score_from_proto(LABEL_FILE)'''
    file_list, score_list = label_util.load_gt_sep(LABEL_FILE)
    file_list, score_list = shuffleL1L2(file_list, score_list)
    #np.random.shuffle(file_list)
    #np.random.shuffle(score_list)

    for path, y in zip(file_list, score_list):
        print 'processing ' + path
        img = load_image(ROOT + os.sep + path)
        #img = img_mani.filter_img(img)
        if img is False:
            continue
        imgs = augment_img(img)

        dice = np.random.rand()
        if dice > 0.95:
            save_batch(os.path.split(path)[1], imgs, additional_name_info=str(y))

        imgs = np.swapaxes(imgs, 1, 3)
        imgs = np.swapaxes(imgs, 2, 3)

        y = np.repeat(y, imgs.shape[0], axis=0)
        Xs = np.vstack([Xs, imgs])
        ys = np.hstack([ys, y])
        count += 1
        if Xs.shape[0] >= HD5SIZE:
            write_XY_h5(ROOT + os.sep + PHASE + str(count) + '.h5', Xs, ys, shuffle=True)
            h5name_record.append(ROOT + os.sep + PHASE + str(count) + '.h5')
            Xs, ys = get_empty_XY()

    # in case number of last batch of X,y is smaller than HD5SIZE
    write_XY_h5(ROOT + os.sep + PHASE + str(count) + '.h5', Xs, ys, shuffle=True)
    h5name_record.append(ROOT + os.sep + PHASE + str(count) + '.h5')
    write_h5names(ROOT + os.sep + PHASE + '_h5_list.txt', h5name_record)

if __name__ == '__main__':
    convert2h5data()