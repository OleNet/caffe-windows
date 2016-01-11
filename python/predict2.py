import caffe
import numpy as np
import os
import glob
from scipy import stats
import label_file_util as label_util
import matplotlib.pyplot as plt


#caffe_root = './caffe'
NET_FILE = 'D:/Project/caffe-windows-master/models/bvlc_alexnet-sur/deploy.prototxt'
PARAM_FILE = 'D:/Project/caffe-windows-master/models/bvlc_alexnet-sur/alexnet_train201601102248_iter_1000.caffemodel'
#img_path = 'D:/Project/caffe-windows-master/data/Blur1000/test/132.BMP'
IMAGE_ROOT = 'D:/Project/caffe-windows-master/data/Blur1500/test/'
GROUND_TRUTH = 'D:/Project/caffe-windows-master/data/Blur1500/test.proto'
SZ = 170

###################################set mode
caffe.set_mode_gpu()
net = caffe.Net(NET_FILE, PARAM_FILE, caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
#transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
#transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

def predict(image_path):
    net.blobs['data'].reshape(1, 3, SZ, SZ)
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image_path))
    out = net.forward()
    return np.squeeze(out.values())


def predict_batch(image_dir):
    name_list = glob.glob(image_dir + os.sep + '*.bmp')
    score_record = {}

    count = 0
    for path in name_list:
        count += 1
        if count > 10 :
            pass
            #break
        print 'predicting {}'.format(path)
        name = os.path.split(path)[1]
        score_record[name] = predict(path)
    return score_record


def intersect(dict1, dict2):
    if len(dict1) < len(dict2):
      inter = dict.fromkeys([x for x in dict1 if x in dict2])
    else:
      inter = dict.fromkeys([x for x in dict2 if x in dict1])
    v1 = []
    v2 = []
    for key in inter:
        v1.append(float(dict1[key]))
        v2.append(dict2[key])
    return [np.array(v1), np.array(v2)]


def test_case1():
    dict1={'a' : 1, 'b' : 2}
    dict2={'c' : 1, 'b' : 2}
    v1, v2 = intersect(dict1, dict2)
    print stats.spearmanr(v1, v2)[0]


if __name__ == '__main__':
    score = predict_batch(IMAGE_ROOT)
    #gt = label_util.load_groundtruth(GROUND_TRUTH)
    gt = label_util.load_gt_dict(GROUND_TRUTH, split_name=True)
    v1, v2 = intersect(score, gt)
    sorted_ind = np.argsort(v2)
    v1 = v1[sorted_ind]
    v2 = v2[sorted_ind]

    [rho, p_value] = stats.spearmanr(v1, v2)
    #plt.figure()
    score_handle, = plt.plot(v1, 'mo')
    gt_handle,  = plt.plot(v2, 'ko')
    plt.title('srocc: {}'.format(p_value))
    plt.legend([score_handle, gt_handle], ['pred', 'ground truth'])
    plt.ylim(-0.1, 1.1)
    plt.xlabel('#image')
    plt.ylabel('score and gt')
    plt.show()
    #os.system('pause')

