import caffe
import numpy as np
import skimage.io as skio
import img_mani

caffe_root = './caffe'

NET_FILE = 'D:/Project/caffe-windows-master/models/bvlc_alexnet-sur/train_val-deploy.prototxt'
PARAM_FILE = 'D:/Project/caffe-windows-master/models/bvlc_alexnet-sur/alexnet_train201601102248_iter_25000.caffemodel'
img_path = 'D:/Project/caffe-windows-master/data/Blur1000/test/000000000217.BMP'
SZ = 170


caffe.set_mode_gpu()

net = caffe.Net(NET_FILE, PARAM_FILE, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))


def predict(image_path):
    net.blobs['data'].reshape(1, 3, SZ, SZ)
    img = skio.imread(image_path)

    net.blobs['data'].data[...] = transformer.preprocess('data', img/255.0)
    out = net.forward()
    print np.squeeze(out.values())

    img = img_mani.filter_img(img)
    skio.imshow(img)
    skio.show()
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    print np.squeeze(out.values())
    return np.squeeze(out.values())


def caffe_predict(path):
        input_image = caffe.io.load_image(path)
        print path
        print input_image

        prediction = net.predict([input_image])

        print prediction
        print "----------"

        print 'prediction shape:', prediction[0].shape
        print 'predicted class:', prediction[0].argmax()


        proba = prediction[0][prediction[0].argmax()]
        ind = prediction[0].argsort()[-5:][::-1] # top-5 predictions

        return prediction[0].argmax(), proba, ind

if __name__ == '__main__':
    print predict(img_path)
    #caffe_predict(img_path)