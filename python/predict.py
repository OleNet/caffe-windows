import caffe
import numpy as np


caffe_root = './caffe'

NET_FILE = 'D:/Project/caffe-windows-master/models/bvlc_alexnet-sur/train_val.prototxt'
PARAM_FILE = 'D:/Project/caffe-windows-master/models/bvlc_alexnet-sur/alexnet_train201601102248_iter_194000.caffemodel'
#img_path = 'D:/Project/caffe-windows-master/data/Blur1000/test/000000000217.BMP'

caffe.set_mode_gpu()

net = caffe.Classifier(NET_FILE, PARAM_FILE,
               #channel_swap=(2,1,0),
               channel_swap=(2,0,1),
               raw_scale=255,
               image_dims=(170, 170))

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
    caffe_predict(img_path)